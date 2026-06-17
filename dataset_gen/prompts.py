from typing import Dict, List, Any


def build_fact_generation_prompt(user_data: Dict[str, Any], topics: List[Dict]) -> str:
    """
    Build prompt for generating subtopic facts from CSV user data

    Args:
        user_data: Dictionary containing CSV row data for a user
        topics: List of topic dictionaries

    Returns:
        Formatted prompt string
    """
    # Extract key fields with defaults
    occupation = user_data.get("Occupation (O*NET-SOC Title)", "Unknown")
    education = user_data.get("Education", "Unknown")
    experience = user_data.get("Experience", "Unknown")
    age = user_data.get("Age", "Unknown")
    llm_familiarity = user_data.get("LLM Familiarity", "Unknown")
    llm_use_in_work = user_data.get("LLM Use in Work", "Unknown")

    # Rich text fields
    work_description = user_data.get("Work Description", "")
    work_tasks = user_data.get("Work Tasks", "")
    work_tools = user_data.get("Work Tools", "")
    work_processes = user_data.get("Work Processes", "")
    ai_envision = user_data.get("AI Envision", "")

    # Build subtopics list for reference
    subtopics_text = ""
    for topic in topics:
        subtopics_text += f"\n## {topic['topic']}\n"
        for i, subtopic in enumerate(topic['subtopics'], 1):
            subtopic_id = f"{topics.index(topic) + 1}.{i}"
            subtopics_text += f"{subtopic_id}. {subtopic}\n"

    prompt = f"""You are creating a synthetic persona profile for benchmarking an AI interviewer system.

Given the following data about a real worker, generate 3-5 factual bullet points for each of the 48 interview subtopics listed below.

## User Data:
- **Occupation**: {occupation}
- **Education**: {education}
- **Years of Experience**: {experience}
- **Age**: {age}
- **LLM Familiarity**: {llm_familiarity}
- **LLM Use in Work**: {llm_use_in_work}

### Work Description:
{work_description if work_description else "Not provided"}

### Work Tasks:
{work_tasks if work_tasks else "Not provided"}

### Work Tools:
{work_tools if work_tools else "Not provided"}

### Work Processes:
{work_processes if work_processes else "Not provided"}

### AI Vision for Future:
{ai_envision if ai_envision else "Not provided"}

## Instructions:
1. Generate 3-5 specific, factual bullet points for EACH of the 48 subtopics below
2. Base facts on the provided user data whenever possible
3. When user data is sparse for a subtopic, infer realistic details consistent with:
   - The person's occupation and role
   - Their education and experience level
   - Industry norms for similar professionals
4. Maintain GLOBAL CONSISTENCY across all subtopics:
   - No contradictions (e.g., if they use LLMs daily, don't say they've never used AI tools)
   - Ensure personality, work style, and attitudes are coherent
   - Timeline and career progression should be logical
5. Create SPECIFIC, VERIFIABLE facts, not generic statements
   - BAD: "Uses various tools for work"
   - GOOD: "Uses Salesforce CRM daily to track customer interactions and manage sales pipeline"
6. Add depth where appropriate (4-5 facts for core work/AI topics, 3 for background/demographic topics)

## 48 Interview Subtopics:
{subtopics_text}

## Output Format:
Return a JSON array matching this exact structure:

```json
[
  {{
    "topic": "Introduction & Background",
    "subtopics": [
      {{
        "subtopic_id": "1.1",
        "subtopic_description": "Educational background or training",
        "notes": [
          "Holds a {education} degree",
          "Completed degree in [field related to occupation]",
          "Additional certifications or training relevant to role"
        ]
      }},
      ...
    ]
  }},
  ...
]
```

CRITICAL: Include ALL 48 subtopics with 3-5 notes each. Ensure no contradictions across subtopics.
"""
    return prompt

def build_distractor_facts_prompt(occupation: str, core_facts: List[str]) -> str:
    """
    Build prompt for generating creative distractor facts

    Args:
        occupation: User's occupation
        core_facts: List of core facts from topics_filled.json

    Returns:
        Formatted prompt string
    """
    # Show first 10 core facts as examples for context
    sample_facts = "\n".join(f"- {fact}" for fact in core_facts[:10])

    prompt = f"""Generate 40-50 creative, fun biographical facts about this {occupation}.

These facts should add realistic personal details but be UNRELATED to the core work/AI interview topics.

## Examples of Good Distractor Facts:
- Brings homemade cookies to team meetings every Friday
- Office plant is named Gerald and receives daily motivational talks
- Collects vintage coffee mugs from client cities visited
- Has a standing desk that's never been lowered in 3 years
- Organizes monthly book club focusing on sci-fi novels
- Learned to play ukulele during pandemic lockdown
- Commutes by bicycle even in winter
- Office is decorated with posters from 1990s tech conferences
- Always eats lunch at exactly 12:30 PM
- Has a collection of rubber ducks on desk for debugging companionship

## What Makes Good Distractor Facts:
1. **Creative and specific**: Not generic, has personality
2. **Realistic**: Feels like something a real person would do
3. **Grounded in occupation context**: Makes sense for this profession
4. **Personal quirks, hobbies, habits**: Office details, food preferences, anecdotes, collections
5. **Fun and memorable**: Adds human color to the profile

## What to AVOID (these overlap with core interview topics):
- Direct mentions of specific AI tools or usage patterns
- Details about primary job responsibilities or decision-making
- Technical skills, certifications, or work processes
- Team size, company info, or organizational structure
- Opinions on AI's impact on work

## Core Facts to Avoid Contradicting:
{sample_facts}
... (and {len(core_facts) - 10} more core facts)

## Task:
Generate 40-50 creative distractor facts for this {occupation}.
BE CREATIVE! Make them fun, realistic, and memorable.
Ensure they DON'T contradict any core facts above.

## Output Format:
Return a JSON array of 40-50 strings:
```json
[
  "Distractor fact 1",
  "Distractor fact 2",
  ...
]
```

Generate distractor facts now:
"""
    return prompt
