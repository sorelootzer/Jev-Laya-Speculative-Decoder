import time
import json
from getpass import getpass
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MAX_CHOICES = 24


class JEVSelector:

    def __init__(self, k, branching_factor=2, model_name="jev-latest",
                 api_key=None, timeout=30):
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")
        if not isinstance(branching_factor, int) or branching_factor < 1:
            raise ValueError("branching_factor must be a positive integer.")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.k = k
        self.branching_factor = branching_factor
        self.expected_draft_count = branching_factor ** k
        self.model_name = model_name
        self.timeout = timeout
        # Prompt once per process; retain only in memory, never in results.
        self._api_key = api_key if api_key is not None else getpass("Enter JEV API key: ").strip()
        if not self._api_key:
            raise ValueError("JEV API key cannot be empty")
        self.load_time = 0.0  # Remote service: no local model initialization.

    def predict(self, state, questions):
        """Call the documented JEV choice API; do not retry paid requests."""
        if not isinstance(questions, dict) or not questions:
            raise ValueError("JEV questions must be a nonempty mapping")
        if len(questions) > 20:
            raise ValueError("JEV supports at most 20 questions per request")
        for question in questions.values():
            if question.get("type") == "choice":
                criteria = question.get("criteria")
                if not isinstance(criteria, dict) or not 2 <= len(criteria) <= MAX_CHOICES:
                    raise ValueError("JEV choice questions require 2–24 candidates")
        body = json.dumps({"model": self.model_name, "state": state,
                           "questions": questions}).encode("utf-8")
        request = Request(
            JEV_ENDPOINT, data=body, method="POST",
            headers={"Authorization": "Bearer " + self._api_key,
                     "Content-Type": "application/json", "Accept": "application/json"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            # Never include response body or request headers (which may contain secrets).
            raise RuntimeError(f"JEV API HTTP {exc.code}; check key, balance, limits, or request") from None
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(f"JEV network/timeout error ({type(exc).__name__})") from None
        except (UnicodeError, json.JSONDecodeError):
            raise RuntimeError("JEV API returned an invalid JSON response") from None
        if not isinstance(result, dict) or not isinstance(result.get("answers"), dict):
            raise RuntimeError("JEV API response has no answers object")
        return result


    # ========================================================
    # VALIDATION
    # ========================================================

    def _validate_drafts(self, drafts):
        """
        Ensure Gemma supplied usable candidate drafts.
        """

        if not drafts:
            raise ValueError(
                "JEV received no drafts."
            )

        if len(drafts) > self.expected_draft_count:
            raise ValueError(
                f"Received {len(drafts)} drafts, but at most "
                f"{self.expected_draft_count} are possible for "
                f"k={self.k}, branching_factor={self.branching_factor}."
            )

        required_fields = {
            "name",
            "token_ids",
            "text"
        }

        names = set()

        for draft in drafts:

            missing = (
                required_fields
                - draft.keys()
            )

            if missing:
                raise ValueError(
                    "Draft is missing required "
                    f"fields: {missing}"
                )

            name = str(
                draft["name"]
            )

            token_count = len(draft["token_ids"])

            if token_count < 1 or token_count > self.k:
                raise ValueError(
                    f"Draft {name} contains {token_count} token IDs; "
                    f"expected between 1 and k={self.k}."
                )

            if len(name) != token_count:
                raise ValueError(
                    f"Draft name {name!r} has length {len(name)}, but "
                    f"the draft contains {token_count} token IDs."
                )

            is_terminal = draft.get("is_terminal", False)

            if token_count < self.k and not is_terminal:
                raise ValueError(
                    f"Draft {name!r} ends before k={self.k} but is not "
                    "marked is_terminal=True."
                )

            valid_branch_digits = {
                str(i)
                for i in range(1, self.branching_factor + 1)
            }

            if any(digit not in valid_branch_digits for digit in name):
                raise ValueError(
                    f"Draft name {name!r} contains a branch outside "
                    f"1..{self.branching_factor}."
                )

            if name in names:
                raise ValueError(
                    "Duplicate draft name "
                    f"received: {name}"
                )

            names.add(name)

        return names


    # ========================================================
    # BUILD LAYA STATE
    # ========================================================

    def _build_state(
        self,
        user_prompt,
        assistant_response
    ):
        """
        JEV receives semantic context rather than
        Gemma's internal chat/control tokens.
        """

        return {
            "user_request":
                user_prompt,

            "assistant_response_so_far":
                assistant_response
        }


    # ========================================================
    # BUILD LAYA QUESTION
    # ========================================================

    def _build_questions(
        self,
        drafts
    ):
        """
        Convert all Gemma candidate drafts into one
        JEV choice question.

        Draft names are dynamic branch paths. For example,
        k=2, b=2 produces 11, 12, 21, 22, while
        k=3, b=2 produces 111 through 222.
        """

        criteria = {}

        for draft in drafts:

            name = str(
                draft["name"]
            )

            # Keep criteria as strings for JEV's choice API. Terminal
            # status describes the option, but does not change Gemma tokens.
            is_terminal = bool(draft.get("is_terminal", draft.get("terminal", False)))
            status = "[COMPLETE: stops after this draft]" if is_terminal else "[CONTINUATION]"
            criteria[name] = f"{draft['text']} {status}"

        return {

            "draft_selection": {

                "type":
                    "choice",

                "instructions": (
                    "Choose the candidate that best satisfies the user's "
                    "request, considering the assistant response so far. "
                    "Prioritize factual and logical correctness and obey "
                    "the requested output format and length. "
                    "A COMPLETE option ends the response immediately; "
                    "prefer it when it already provides a sufficient answer. "
                    "Do not extend an answer merely to add explanation. "
                    "A CONTINUATION option will require further generation. "
                    "Do not choose a complete option if its answer is incorrect."
                ),

                "criteria":
                    criteria
            }
        }


    # ========================================================
    # FIND ORIGINAL GEMMA DRAFT
    # ========================================================

    def _find_draft(
        self,
        drafts,
        selected_name
    ):
        """
        Return the ORIGINAL Gemma draft.

        This is important because it contains the
        exact token IDs generated by Gemma.
        """

        for draft in drafts:

            if (
                str(draft["name"])
                == str(selected_name)
            ):

                return draft

        raise ValueError(
            "JEV selected unknown draft: "
            f"{selected_name}"
        )


    # ========================================================
    # MAIN SELECTION FUNCTION
    # ========================================================

    def select(
        self,
        user_prompt,
        assistant_response,
        drafts
    ):
        """
        Ask JEV to select one of Gemma's drafts.

        Returns:

        Dictionary containing selected_draft, selected_name,
        candidate_evaluations, metadata and raw_result.
        """

        valid_names = (
            self._validate_drafts(
                drafts
            )
        )

        state = self._build_state(
            user_prompt,
            assistant_response
        )

        questions = (
            self._build_questions(
                drafts
            )
        )


        # ====================================================
        # LAYA INFERENCE
        # ====================================================

        start = time.perf_counter()

        result = self.predict(
            state,
            questions
        )

        end = time.perf_counter()

        decision_time = (
            end - start
        )


        # ====================================================
        # VALIDATE RESULT STRUCTURE
        # ====================================================

        if (
            "answers"
            not in result
        ):
            raise RuntimeError(
                "JEV result contains no "
                "'answers' field."
            )

        answers = result[
            "answers"
        ]

        if (
            "draft_selection"
            not in answers
        ):
            raise RuntimeError(
                "JEV result contains no "
                "'draft_selection' answer."
            )

        answer = answers[
            "draft_selection"
        ]

        if (
            "choice"
            not in answer
        ):
            raise RuntimeError(
                "JEV did not return a choice."
            )

        selected_name = str(
            answer["choice"]
        )

        if (
            selected_name
            not in valid_names
        ):
            raise RuntimeError(
                "JEV selected an invalid "
                f"draft: {selected_name}"
            )


        # ====================================================
        # GET ORIGINAL GEMMA DRAFT
        # ====================================================

        selected_draft = (
            self._find_draft(
                drafts,
                selected_name
            )
        )


        # ====================================================
        # EXTRACT LAYA INFORMATION
        # ====================================================

        probabilities = (
            answer.get(
                "probabilities",
                {}
            )
        )

        confidence = (
            answer.get(
                "confidence"
            )
        )

        answer_confidence = (
            answer.get(
                "answer_confidence"
            )
        )

        action = (
            answer.get(
                "action",
                {}
            )
        )

        usage = (
            result.get(
                "usage",
                {}
            )
        )


        # ====================================================
        # GEMMA / LAYA COMPARISON
        # ====================================================

        gemma_best = max(
            drafts,

            key=lambda draft:
                draft.get(
                    "joint_log_probability",
                    float("-inf")
                )
        )

        gemma_best_name = str(
            gemma_best["name"]
        )

        agreement = (
            gemma_best_name
            == selected_name
        )


        # ====================================================
        # METADATA
        # ====================================================

        metadata = {

            # -----------------------------------------------
            # Selection
            # -----------------------------------------------

            "selected_name":
                selected_name,

            "selected_text":
                selected_draft["text"],

            "draft_length":
                self.k,

            "branching_factor":
                self.branching_factor,

            "draft_count":
                len(drafts),

            "expected_draft_count":
                self.expected_draft_count,

            "terminal_draft_count":
                sum(
                    1
                    for draft in drafts
                    if draft.get("is_terminal", False)
                ),


            # -----------------------------------------------
            # JEV probabilities
            # -----------------------------------------------

            "probabilities":
                probabilities,

            "confidence":
                confidence,

            "answer_confidence":
                answer_confidence if answer_confidence is not None else probabilities.get(selected_name),

            "action":
                action,


            # -----------------------------------------------
            # Performance
            # -----------------------------------------------

            "decision_seconds":
                decision_time,

            "load_seconds":
                self.load_time,


            # -----------------------------------------------
            # Usage
            # -----------------------------------------------

            "input_tokens":
                usage.get(
                    "input_tokens"
                ),

            "state_tokens":
                usage.get(
                    "state_tokens"
                ),

            "state_tokens_dropped":
                usage.get(
                    "state_tokens_dropped"
                ),

            "truncated":
                usage.get(
                    "truncated"
                ),

            "truncated_questions":
                usage.get(
                    "truncated_questions",
                    []
                ),


            # -----------------------------------------------
            # Gemma comparison
            # -----------------------------------------------

            "gemma_best_name":
                gemma_best_name,

            "gemma_best_text":
                gemma_best["text"],

            "gemma_laya_agree":
                agreement,
            "selector_backend": "jev",
            "model_version": result.get("model_version", result.get("model", self.model_name))
        }


        # Complete structured selector output; no additional JEV inference.
        return {
            "selected_draft": selected_draft,
            "selected_name": selected_name,
            "candidate_evaluations": {
                str(draft["name"]): {
                    "draft": draft,
                    "gemma_probability": draft.get("joint_probability"),
                    "laya_probability": probabilities.get(str(draft["name"])),
                    "jev_probability": probabilities.get(str(draft["name"])),
                }
                for draft in drafts
            },
            "metadata": metadata,
            "raw_result": result,
            "selector_backend": "jev",
        }