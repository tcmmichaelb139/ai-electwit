import json_repair

test_str = """
lajsdlkfjasd;f
asd;lkfja;slkdjf
asdfasdf
asdf
as
df
asd
fa
sdf
[
    
        {
        
            "id": "1",
            "name": "Alice",
            "day": 1,
            "hour": 10,
            "content": "This is a test comment.",
            "replies": [],
            "likes": 0
            }
            ]"""

matches = json_repair.loads(test_str)
print(matches)
