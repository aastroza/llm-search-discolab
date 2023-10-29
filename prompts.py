from jinja2 import Template

DOCUMENT_QA_PROMPT_TEMPLATE = Template(
    """
    Please provide an answer based solely on the provided sources. When referencing information from a source, cite the appropriate source(s) using their corresponding numbers. Every answer should include at least one source citation. Only cite a source when you are explicitly referencing it. If none of the sources are helpful, you should indicate that. For example:
    Source 1:
    The sky is red in the evening and blue in the morning.
    Source 2:
    Water is wet when the sky is red.
    Query: When is water wet?
    Answer: Water will be wet when the sky is red [2], which occurs in the evening [1].
    Now it's your turn. Below are several numbered sources of information:
    ------
    {{context}}
    ------
    Query: {{query}}
    Answer: 
    """
)

FINAL_RESPONSE_PROMPT_TEMPLATE = Template(
    """
    You are a Constitutional Lawyer. You are asked to give a brief response about 
    the diferences of two constitutions about this topic: {{query}}.

    The first constitution is the current one, and the second one is a proposed one.
    Always refer to the first constitution as {{document1_title}} and the second one as {{document2_title}}.

    The first constitution says the following about the topic: {{first_response}}.
    The second constitution says the following about the topic: {{second_response}}.

    Please detail the differences between the two constitutions about this topic.
    Please be concise and respond in spanish.
    """
)