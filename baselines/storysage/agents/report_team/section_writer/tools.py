from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain_core.callbacks.manager import CallbackManagerForToolRun
from langchain_core.tools import BaseTool, ToolException


from content.report.report import Report


class UpdateSectionInput(BaseModel):
    path: Optional[str] = Field(description="Original full path to the section to update. Optional if you want to update the title instead.", default=None)
    title: Optional[str] = Field(description="Title of the section to update. Optional if you want to update the content instead.", default=None)
    content: str = Field(description="Updated content for the section")
    new_title: Optional[str] = Field(description="Updated title for the section. Only provide if you want to change the title.", default=None)

class UpdateSection(BaseTool):
    """Tool for updating existing sections."""
    name: str = "update_section"
    description: str = "Update content of an existing section"
    args_schema: Type[BaseModel] = UpdateSectionInput
    report: Report

    async def _run(
        self,
        content: str,
        path: Optional[str] = None,
        title: Optional[str] = None,
        new_title: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        section = await self.report.update_section(
            path=path,
            title=title,
            content=content,
            new_title=new_title
        )
        if not section:
            identifier = path if path else title
            raise ToolException(f"Section '{identifier}' not found")
        return f"Successfully updated section '{path if path else title}'"


class AddSectionInput(BaseModel):
    path: str = Field(
        description="Full path to the new section (e.g., '1 Early Life/1.1 Childhood')")
    content: str = Field(description="Content of the new section")


class AddSection(BaseTool):
    """Tool for adding new sections."""
    name: str = "add_section"
    description: str = "Add a new section to the report"
    args_schema: Type[BaseModel] = AddSectionInput
    report: Report

    async def _run(self, path: str, content: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            await self.report.add_section(path, content)
            return f"Successfully added section at path '{path}'"
        except Exception as e:
            raise ToolException(f"Error adding section: {e}")
        
class AddSubSubSectionInput(BaseModel):
    section_title: str = Field(description="The title of the top-level section (e.g., '1 Role & Background')")
    subsection_title: str = Field(description="The title of the subsection (e.g., '1.1 Job Title & Experience')")
    subsubsection_title: str = Field(description="The title of the new sub-subsection to add (e.g., '1.1.1 Years in Role')")
    content: str = Field(description="Content of the new sub-subsection")


class AddSubSubSection(BaseTool):
    """Tool for adding new sections."""
    name: str = "add_sub_sub_section"
    description: str = "Add a new sub-sub-section to the report"
    args_schema: Type[BaseModel] = AddSubSubSectionInput
    report: Report

    async def _run(self, section_title: str, subsection_title: str, subsubsection_title: str, content: str, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            await self.report.add_sub_sub_section(section_title, subsection_title, subsubsection_title, content)
            return f"Successfully added section at path '{section_title}/{subsection_title}/{subsubsection_title}'"
        except Exception as e:
            raise ToolException(f"Error adding section: {e}")
