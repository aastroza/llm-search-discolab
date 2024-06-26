from jinja2 import Template

DOCUMENT_QA_SYSTEM_PROMPT = Template(
"""
Please respond in {{language}}.
Please provide an answer based solely on the provided sources. When referencing information from a source, cite the appropriate source(s) using their corresponding numbers enclosed within square brackets. Every answer should include at least one source citation. Only cite a source when you are explicitly referencing it. If none of the sources are helpful, you should indicate that. For example:
Source 1:
The sky is red in the evening and blue in the morning.
Source 2:
Water is wet when the sky is red.
Query: When is water wet?
Answer: Water will be wet when the sky is red [2], which occurs in the evening [1].
"""
)

DOCUMENT_QA_USER_PROMPT_TEMPLATE = Template(
"""
Now it's your turn. Below are several numbered sources of information:
------
{{context}}
------
Query: {{query}}
Answer: 
"""
)

DOCUMENT_QA_REFINE_SYSTEM_PROMPT = Template(
"""
Please respond in {{language}}.
Please provide an answer based solely on the provided sources. When referencing information from a source, cite the appropriate source(s) using their corresponding numbers enclosed within square brackets. Every answer should include at least one source citation. Only cite a source when you are explicitly referencing it. If none of the sources are helpful, you should indicate that. For example:
Source 1:
The sky is red in the evening and blue in the morning.
Source 2:
Water is wet when the sky is red.
Query: When is water wet?
Answer: Water will be wet when the sky is red [2], which occurs in the evening [1].
"""
)

DOCUMENT_QA_REFINE_USER_PROMPT_TEMPLATE = Template(
"""
Now it's your turn. We have provided an existing answer: {{existing_answer}}. Below are several numbered sources of information. Use them to refine the existing answer. If the provided sources are not helpful, you will repeat the existing answer.
Begin refining!
------
{{context}}
------
Query: {{query}}
Answer: 
"""
)

FINAL_RESPONSE_SYSTEM_PROMPT_TEMPLATE = Template(
"""
You are a Constitutional Lawyer. You are asked to give a brief response about 
the differences of two constitutions.

The first constitution is the current one, and the second one is a proposed one.
Always refer to the first constitution as "{{document1_title}}" and the second one as "{{document2_title}}".

If none of the constitutions are informative about the topic, you should indicate that.

Please respond in {{language}}.
"""
)

FINAL_RESPONSE_USER_PROMPT_TEMPLATE = Template(
"""
Please explain the differences of two constitutions about this topic: {{query}}.

The first constitution says the following about the topic: {{first_response}}.
The second constitution says the following about the topic: {{second_response}}.

Please detail the differences between the two constitutions about this topic.
Please be concise.
"""
)