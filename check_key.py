"""Check the API key in .env without printing it.   Run:  python check_key.py"""
import hashlib

from dotenv import dotenv_values

OLD_KEY_FINGERPRINT = "cf3d155d"  # the key OpenAI rejected with a 401

key = dotenv_values(".env").get("OPENAI_API_KEY") or ""
fingerprint = hashlib.sha256(key.encode()).hexdigest()[:8] if key else "-"
print("key length     :", len(key))
print("fingerprint    :", fingerprint)
print("ends with      :", repr(key[-4:]) if key else "-")

if not key:
    raise SystemExit("No OPENAI_API_KEY found in .env")
if fingerprint == OLD_KEY_FINGERPRINT:
    raise SystemExit("This is still the OLD key that OpenAI rejected. Replace the line in .env and press Ctrl+S.")

from evalkit.llm import client, friendly_error

try:
    n = len(client().models.list().data)
    print("RESULT         : KEY OK, OpenAI accepted it (", n, "models visible )")
except Exception as e:
    print("RESULT         : KEY FAILED -", friendly_error(e))
