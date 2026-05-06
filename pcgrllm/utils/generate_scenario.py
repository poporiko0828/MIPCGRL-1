import itertools
import json
import random

# Update prompts and base prompt to English
base_prompt_english = (
    "The 'Player' needs to obtain the Key and escape through the 'Door'."
    "To pick up the key, the player must encounter one of monsters: {monsters}."
    "The players can figure out all of monsters when they play the level several times."
)

monsters = [["BAT"], ["BAT", "SCORPION"], ["BAT", "SCORPION", "SPIDER"]]


final_result_with_english_prompts = {
    "base_prompt": base_prompt_english,
    "scenarios": {}
}

for idx, (monster_group) in enumerate(monsters, start=1):
    # Generate the prompt
    prompt = base_prompt_english.format(
        monsters=", ".join(monster_group),
    )

    final_result_with_english_prompts["scenarios"][idx] = {
        "prompt": prompt,
        "important_tiles": monster_group,
    }

# Convert to JSON format
final_result_with_english_prompts_json = json.dumps(final_result_with_english_prompts, indent=4)

# Save to JSON
with open("scenario_preset.json", "w") as f:
    f.write(final_result_with_english_prompts_json)