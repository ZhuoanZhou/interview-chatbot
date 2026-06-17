# Define perspective templates

THIRD_PERSON_INSTRUCTION = """
Perspective Requirements:
- ALWAYS write in third-person perspective
- Use subject's name and appropriate pronouns (he/she/they)
- Never use first-person ("I", "my", "we", "our")
- Examples:
  ✓ "Margaret graduated from Harvard in 1985"
  ✗ "I graduated from Harvard in 1985"
  ✓ "Her father taught her how to fish"
  ✗ "My father taught me how to fish"
"""

REPORT_STYLE_PLANNER_INSTRUCTION = """
As an professional report planner:
- Organize content with scholarly precision into structured sections aligned with the given fixed topics
- Focus on factual accuracy and documentation
- Quantify whenever possible, usually questions with rubric contain quantified and therefore more objective answer.
- Look for historical context and influences
- Plan sub-sections that analyze rather than just narrate
- Structure content with clear supporting evidence
- Ask follow-up questions to verify facts and details
"""

REPORT_STYLE_WRITER_INSTRUCTION = """
Writing Style Requirements:
{THIRD_PERSON_INSTRUCTION}

Additional Style Elements:
- Maintain formal, scholarly tone
- Use precise, specific language
- Provide context and analysis
- Focus on factual presentation, and quantify whenever possible.
- Include relevant background information
- Use objective, analytical language
- Support statements with evidence
"""