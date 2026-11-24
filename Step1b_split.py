from pathlib import Path
path=Path("DATA/CORPUS/www-myvi-in-help-support-faqs-how-to-setup-default-home-screen-in-vi-app.txt")
text=path.read_text(encoding="utf-8")
print(text)
print("length of text:",len(text))

first_line, seperator, body= text.partition("\n")
body=body.strip()

print("first line:", first_line)
print("separator:", seperator)
print("body:", body)    

source = first_line.replace("SOURCE:"," ").strip()
print("First line URL:",source)
print("Body",body[:80])


