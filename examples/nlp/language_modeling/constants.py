TASKS = {"qasper":
                {"response": "Answer:",
                "tokens_to_generate": 30,
                "subset": "Qasper/test.jsonl"},
        "narrative_qa":
                {"response": "Answer:",
                "tokens_to_generate": 29,
                "subset": "NarrativeQA/test.jsonl"},
        "quality":
                {"response": "Answer:",
                "tokens_to_generate": 128,
                "subset": "QuALITY/validation.jsonl"},
        "musique":
                {"response": "Answer:",
                "tokens_to_generate": 128,
                "subset": "MuSiQue/validation.jsonl"},
        "space_digest":
                {"response": "Percentage of Positive Reviews:",
                "tokens_to_generate": 128,
                "subset": "SpaceDigest/test.jsonl"},
        "book_sum_sort":
                {"response": "Summary IDs in Correct Order:",
                "tokens_to_generate": 128,
                "subset": "BookSumSort/validation.jsonl"},
        "gov_report":
                {"response": "Summary:",
                "tokens_to_generate": 128,
                "subset": "GovReport/test.jsonl"},
        "summ_screen_fd":
                {"response": "Summary:",
                "tokens_to_generate": 128,
                "subset": "SummScreenFD/test.jsonl"},
        "qmsum":
                {"response": "Answer:",
                "tokens_to_generate": 128,
                "subset": "QMSum/test.jsonl"},
        "squality":
                {"response": "Answer:",
                "tokens_to_generate": 128,
                "subset": "SQuALITY/validation.jsonl"}
        }

EXTRA_TOKENS = 10
