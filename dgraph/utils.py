def get_uid_command(id: str) -> str:
    return f"""
    {{
        query(func: eq(id, {id})){{
            uid
        }}
    }}
"""

def delete_uid_command(id: str) -> str:
    return f"""
    {{
        "delete": [
            {{
                "uid": "{id}"
            }}
        ]
    }}
"""

