selected_entity_names = ["str", "hello", "hehe"]

source_text = ' OR '.join([f'eq(source, "{name}")' for name in selected_entity_names])

print(source_text)