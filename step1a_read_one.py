from pathlib import Path
path=Path("DATA/CORPUS/www-myvi-in-help-support-faqs-how-to-setup-default-home-screen-in-vi-app.txt")
text=path.read_text(encoding="utf-8")
print(text)
print("Number of characters in text is:", len(text))