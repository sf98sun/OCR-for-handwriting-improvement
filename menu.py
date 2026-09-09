# cli.py
from InquirerPy import inquirer
from quality_check import run as run_quality_check
from preprocessor import run as run_processor

def main():
    while True:
        choice = inquirer.select(
            message="Select a function to run:",
            choices=[
                "Photo Preprocessor",
                "Quality Check",
                "Exit"
            ],
        ).execute()

        if choice == "Quality Check":
            quality_check()

        elif choice == "Photo Preprocessor":
            photo_preprocessor()

        else:
            print("Goodbye.")
            break


def quality_check():
    while True:
        photo = inquirer.text(
            message="Path to the processed image:",
            validate=lambda x: len(x.strip()) > 0 or "Cannot be empty"
        ).execute()

        template = inquirer.text(
            message="Path to the template image:",
            validate=lambda x: len(x.strip()) > 0 or "Cannot be empty"
        ).execute()

        confirm = inquirer.select(
            message="Proceed or go back?",
            choices=["Run", "Re-enter inputs", "Back to main menu"],
        ).execute()

        if confirm == "Run":
            run_quality_check(template, photo)
            break
        elif confirm == "Re-enter inputs":
            continue
        else:
            return


def photo_preprocessor():
    while True:
        template = inquirer.text(
            message="Name of the template image(PNG):",
            validate=lambda x: len(x.strip()) > 0 or "Cannot be empty"
        ).execute()

        photo = inquirer.text(
            message="Name of the raw photo(PNG):",
            validate=lambda x: len(x.strip()) > 0 or "Cannot be empty"
        ).execute()

        confirm = inquirer.select(
            message="Proceed or go back?",
            choices=["Run", "Re-enter inputs", "Back to main menu"],
        ).execute()

        if confirm == "Run":
            run_processor(template, photo)
            break
        elif confirm == "Re-enter inputs":
            continue
        else:
            return


if __name__ == "__main__":
    main()
