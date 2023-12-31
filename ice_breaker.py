from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from output_parsers import person_intel_parser, PersonIntel


def ice_break(name:str)-> tuple[PersonIntel,str]:

    linkedin_profile_url = linkedin_lookup_agent(name = name)

    summary_template = """
       given the information {information} about a person I want you to create:
        1. a short summary
        2. two interesting facts about them
        3. a topic that may interest them
        4. 2 create Ice Breakers to Open a Conversaion with them
        \n{format_instructions}
    """

    summary_prompt_template = PromptTemplate(
        input_variables = ["information"],
        template = summary_template,
        partial_variables = {"format_instructions":person_intel_parser.get_format_instructions()}
    )

    llm = ChatOpenAI(temperature = 1, model_name = "gpt-3.5-turbo")

    chain = LLMChain(llm = llm, prompt = summary_prompt_template)

    linkedin_data = scrape_linkedin_profile()

    result = chain.run(information = linkedin_data)
    print(result)
    return person_intel_parser.parse(result), linkedin_data.get("profile_pic_url")

if __name__ == '__main__':
    print("Hello LangChain!")
    result = ice_break(name = "Harrison Chase")
