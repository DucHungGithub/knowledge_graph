GENERATE_DOMAIN_PROMPT = """
You are an intelligent assistant that helps a human to analyze the information in a text document.
Given a sample text, help the user by assigning a descriptive domain that summarizes what the text is about.
Example domains are: "Social studies", "Algorithmic analysis", "Medical science", among others.

Text: {input_text}
Domain:"""