Translate the Bengali file at $ARGUMENTS to English.

Steps:
1. Verify the venv is active: `source .venv/bin/activate`
2. Run: `bn-translate --input $ARGUMENTS --output ${ARGUMENTS%.bn.txt}.en.txt --model nllb-600M --device auto`
3. Report the output file path and first 5 lines of the translation.

If the user specifies `--model indicTrans2-1B`, use that instead.
If the input path has no `.bn.txt` extension, derive a sensible output name.
