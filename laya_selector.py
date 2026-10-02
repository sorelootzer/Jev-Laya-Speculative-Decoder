import time
import laya

class LayaSelector:

    def __init__(
        self,
        model_name="convaiinnovations/laya",
        k=2,
        branching_factor=2
    ):
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")

        if not isinstance(branching_factor, int) or branching_factor < 1:
            raise ValueError("branching_factor must be a positive integer.")

        self.k = k
        self.branching_factor = branching_factor
        self.expected_draft_count = branching_factor ** k

        print("Loading Laya...")

        start = time.perf_counter()

        self.agent = laya.load(
            model_name
        )

        end = time.perf_counter()

        self.load_time = end - start

        print(
            f"Laya loaded in "
            f"{self.load_time:.3f} s"
        )


    # ========================================================
    # VALIDATION
    # ========================================================

    def _validate_drafts(self, drafts):
        """
        Ensure Gemma supplied usable candidate drafts.
        """

        if not drafts:
            raise ValueError(
                "Laya received no drafts."
            )

        if len(drafts) != self.expected_draft_count:
            raise ValueError(
                f"Expected {self.expected_draft_count} drafts "
                f"for k={self.k}, branching_factor={self.branching_factor}, "
                f"but received {len(drafts)}."
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

            if len(draft["token_ids"]) != self.k:
                raise ValueError(
                    f"Draft {name} contains {len(draft['token_ids'])} token IDs; "
                    f"expected exactly k={self.k}."
                )

            if len(name) != self.k:
                raise ValueError(
                    f"Draft name {name!r} has length {len(name)}; "
                    f"expected a {self.k}-level branch path."
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
        Laya receives semantic context rather than
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
        Laya choice question.

        Draft names are dynamic branch paths. For example,
        k=2, b=2 produces 11, 12, 21, 22, while
        k=3, b=2 produces 111 through 222.
        """

        criteria = {}

        for draft in drafts:

            name = str(
                draft["name"]
            )

            criteria[name] = (
                draft["text"]
            )

        return {

            "draft_selection": {

                "type":
                    "choice",

                "instructions": (
                    "Choose the candidate continuation "
                    "that best continues the assistant "
                    "response while correctly and "
                    "coherently answering the user's "
                    "request."
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
            "Laya selected unknown draft: "
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
        Ask Laya to select one of Gemma's drafts.

        Returns:

        selected_draft, metadata
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

        result = self.agent.predict(
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
                "Laya result contains no "
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
                "Laya result contains no "
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
                "Laya did not return a choice."
            )

        selected_name = str(
            answer["choice"]
        )

        if (
            selected_name
            not in valid_names
        ):
            raise RuntimeError(
                "Laya selected an invalid "
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


            # -----------------------------------------------
            # Laya probabilities
            # -----------------------------------------------

            "probabilities":
                probabilities,

            "confidence":
                confidence,

            "answer_confidence":
                answer_confidence,

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
                agreement
        }


        return (
            selected_draft,
            metadata
        )