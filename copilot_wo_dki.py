import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Answer one geologic map question using HIE and PEQA.")
    parser.add_argument("image", type=Path, help="Path to the geologic map image.")
    parser.add_argument("question", help="Question about the map.")
    parser.add_argument("question_type", help="GeoMap-Bench question type, e.g. extracting-sheet_name.")
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")

    from utils import prompt

    if args.question_type not in prompt.question_ability2type:
        parser.error(f"Unknown question type: {args.question_type}")

    from copilot import copilot

    answer = copilot(str(args.image), args.question, args.question_type, copilot_modes=["HIE", "PEQA"])
    if not answer:
        raise SystemExit("No answer returned. Check the model API logs above.")
    print(f"Question: {args.question}\nAnswer: {answer}")


if __name__ == "__main__":
    main()
