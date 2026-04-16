import os
import dotenv
import asyncio
from typing import Any

from llama_index.core.agent.workflow import AgentOutput, ToolCall, ToolCallResult, FunctionAgent, AgentWorkflow
from llama_index.core.prompts import RichPromptTemplate
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.llms.litellm import LiteLLM
from github import Github, Auth

dotenv.load_dotenv()

"""
Please provide the full URL to your recipes-api GitHub repository below.
"""

auth = Auth.Token(os.getenv("GITHUB_TOKEN", "")) if os.getenv("GITHUB_TOKEN") else None
git = Github(auth=auth if auth else None)

repo_url = "https://github.com/manivelpvn/recipes-api.git"
repo_name = repo_url.split('/')[-1].replace('.git', '')
username = repo_url.split('/')[-2]
full_repo_name = f"{username}/{repo_name}"

if git is not None:
    repo = git.get_repo(full_repo_name)

llm = LiteLLM(
    model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)


async def get_pr_details(pr_number: int) -> dict[str, Any]:
    """
    Useful for retrieving details about a specific pull request using PR number and return PR details as a dictionary
    """
    pull_request = repo.get_pull(pr_number)
    commit_shas = []
    commits = pull_request.get_commits()

    for c in commits:
        commit_shas.append(c.sha)

    head_sha = None
    if pull_request.head is not None:
        head_sha = pull_request.head.sha

    return {
        "title": pull_request.title,
        "number": pull_request.number,
        "author": pull_request.user.login,
        "body": pull_request.body or "",
        "diff_url": pull_request.diff_url,
        "state": pull_request.state,
        "head_sha": head_sha,
        "commits": pull_request.commits,
        "commit_shas": commit_shas
    }


async def get_commit_details(sha: str) -> list[dict[str, Any]]:
    """
    Useful for retrieving commit details for a specific commit sha
    """
    commit = repo.get_commit(sha)
    changed_files: list[dict[str, Any]] = []
    for f in commit.files:
        changed_files.append({
            "filename": f.filename,
            "status": f.status,
            "additions": f.additions,
            "deletions": f.deletions,
            "changes": f.changes,
            "patch": f.patch,
        })
    return changed_files


# Added missing tool mentioned in prompt
async def get_file_content(path: str, ref: str) -> str:
    """Fetch raw content of a file."""
    return repo.get_contents(path, ref=ref).decoded_content.decode("utf-8")


async def add_comment_to_state(draft_comment: str, ctx: Context) -> str:
    """
    Adds a draft_comment to the workflow state.
    """
    # This key will now be available to other agents via ctx.get("draft_comment")
    await ctx.store.set("draft_comment", draft_comment)
    return "Successfully added the draft comment to the state."

async def add_final_review_comment_to_state(final_review_comment: str, ctx: Context) -> str:
    """
    Adds a final_review comment to the workflow state.
    """
    await ctx.store.set("final_review", final_review_comment)
    return "Successfully added the final review comment to the state."

async def post_final_review(pr_number: int, final_review_comment: str) -> str:
    """
    Posts the final review comment to the pull request.
    """
    pull_request = repo.get_pull(pr_number)
    try:
        pull_request.create_review(body=final_review_comment, event="COMMENT")
        return "Review comment posted successfully"
    except Exception as e:
        print(f"Failed to post final review: {e}")
        return f"Error posting review comment: {e}"

context_agent = FunctionAgent(
    llm=llm,
    name="ContextAgent",
    description="Gathers all the needed context of the Pull Request.",
    tools=[
        FunctionTool.from_defaults(async_fn=get_pr_details),
        FunctionTool.from_defaults(async_fn=get_commit_details),
        FunctionTool.from_defaults(async_fn=get_file_content),
    ],
    system_prompt="""You are the context gathering agent. When gathering context, you MUST gather \n: 
  - The details: author, title, body, diff_url, state, and head_sha; \n
  - Changed files; \n
  - Any requested for files; \n
Once you gather the requested info, you MUST hand control back to the Commentor Agent. 
    """,
    can_handoff_to=["CommentorAgent"]
)

commentor_agent = FunctionAgent(
    llm=llm,
    name="CommentorAgent",
    description="Uses the context gathered by the context agent to draft a pull review comment comment.",
    tools=[FunctionTool.from_defaults(async_fn=add_comment_to_state),],
    system_prompt="""You are the commentor agent that writes review comments for pull requests as a human reviewer would. \n 
Ensure to do the following for a thorough review: 
 - Request for the PR details, changed files, and any other repo files you may need from the ContextAgent. 
 - Once you have asked for all the needed information, write a good ~200-300 word review in markdown format detailing: \n
    - What is good about the PR? \n
    - Did the author follow ALL contribution rules? What is missing? \n
    - Are there tests for new functionality? If there are new models, are there migrations for them? - use the diff to determine this. \n
    - Are new endpoints documented? - use the diff to determine this. \n 
    - Which lines could be improved upon? Quote these lines and offer suggestions the author could implement. \n
 - If you need any additional details, you must hand off to the Commentor Agent. \n
 - You should directly address the author. So your comments should sound like: 
 "Thanks for fixing this. I think all places where we call quote should be fixed. Can you roll this fix out everywhere?" \n
 - You should set the review comment in the state. \n
 - You must hand off to the ReviewAndPostingAgent once you are done drafting and setting review comment in the state. \n
 """,
    can_handoff_to=["ContextAgent", "ReviewAndPostingAgent"]
)

review_and_poster_agent = FunctionAgent(
    llm=llm,
    name="ReviewAndPostingAgent",
    description="Reviews the draft comment created by commentor agent. If the review pass then posts the final review to pull request, otherwise request commentor agent to create a new review comment.",
    tools=[
        FunctionTool.from_defaults(async_fn=post_final_review),
        FunctionTool.from_defaults(async_fn=add_final_review_comment_to_state),
    ],
    system_prompt="""You are the Review and Posting agent. You must use the CommentorAgent to create a review comment. 
Once a review is generated, you need to run a final check and post it to GitHub.
   - The review must: \n
   - Be a ~200-300 word review in markdown format. \n
   - Specify what is good about the PR: \n
   - Did the author follow ALL contribution rules? What is missing? \n
   - Are there notes on test availability for new functionality? If there are new models, are there migrations for them? \n
   - Are there notes on whether new endpoints were documented? \n
   - Are there suggestions on which lines could be improved upon? Are these lines quoted? \n
 If the review does not meet this criteria, you must ask the CommentorAgent to rewrite and address these concerns. \n
 When you are satisfied, post the review to GitHub.
    """,
    can_handoff_to=[commentor_agent.name]
)

workflow_agent = AgentWorkflow(
    agents=[context_agent, commentor_agent, review_and_poster_agent],
    root_agent=review_and_poster_agent.name,
    initial_state={
        "gathered_contexts": "",
        "review_comment": "",
        "final_review": "",
    },
)

ctx = Context(workflow_agent)


async def main():
    query = input().strip()
    prompt = RichPromptTemplate(query)

    handler = workflow_agent.run(prompt.format(), ctx=ctx)

    current_agent = None
    async for event in handler.stream_events():
        if hasattr(event, "current_agent_name") and event.current_agent_name != current_agent:
            current_agent = event.current_agent_name
            print(f"Current agent: {current_agent}")
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print("\\n\\nFinal response:", event.response.content)
            if event.tool_calls:
                print("Selected tools: ", [call.tool_name for call in event.tool_calls])
        elif isinstance(event, ToolCallResult):
            print(f"Output from tool: {event.tool_output}")
        elif isinstance(event, ToolCall):
            print(f"Calling selected tool: {event.tool_name}, with arguments: {event.tool_kwargs}")


if __name__ == "__main__":
    asyncio.run(main())
    git.close()
