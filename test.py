from electwit.utils import load_prompt

consolidated_diary = load_prompt("consolidation_prompt.txt", "electwit/prompts")

print(
    consolidated_diary.format(
        name="TestAgent",
        role="voter",
        consolidated_diary="Test consolidated diary",
        today_diary="Test today's diary",
    )
)
