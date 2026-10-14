# System prompts

One folder per prompt, one file per version (`v1.md`, `v2.md`, ...). See `pacer/ai/prompts.py`.

Rules:
- **Never edit a version that was used.** A change in behavior is a new file with the next number,
  so every result can be traced to the exact prompt that produced it (`name@vN` is logged).
- The header (`name`, `version`, `description`) must match the folder and file name; loading checks it.
- No real personal data in prompts or examples. Examples use invented names.
