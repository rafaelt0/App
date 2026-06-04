import re
import glob

# Unicode ranges for emojis
emoji_pattern = re.compile(
    "["
    "\U0001f600-\U0001f64f"  # emoticons
    "\U0001f300-\U0001f5ff"  # symbols & pictographs
    "\U0001f680-\U0001f6ff"  # transport & map symbols
    "\U0001e000-\U0001f9ff"  # additional symbols & pictographs
    "\U00002700-\U000027bf"  # dingbats
    "\U00002600-\U000026ff"  # misc symbols
    "\U0001f900-\U0001f9ff"  # supplemental symbols & pictographs
    "\U0001fa70-\U0001faff"  # symbols and pictographs extended-A
    "\U00002300-\U000023ff"  # miscellaneous technical
    "]+", flags=re.UNICODE
)

for filepath in glob.glob("/home/rafael/b3explorer/App/**/*.py", recursive=True):
    print(f"\n--- Checking {filepath} ---")
    with open(filepath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f, 1):
            matches = emoji_pattern.findall(line)
            if matches:
                # Filter out normal punctuation if matched by mistake
                print(f"Line {idx}: {line.strip()} (matches: {matches})")
