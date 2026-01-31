from pydantic import BaseModel

class PageDocument(BaseModel):
    corpus_id:int
    doc_id:str
    page:int
    exact_filename:str
    probable_referenced_years:list
    markdown_content:str
    text_content:str


