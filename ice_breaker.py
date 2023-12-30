from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent

if __name__ == '__main__':
    print("Hello LangChain!")

    linkedin_profile_url = linkedin_lookup_agent(name = "Bill Gates")

    summary_template = """
       given the information {information} about a person I want you to create:
        1. a short summary
        2. two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables = ["information"], template = summary_template
    )

    llm = ChatOpenAI(temperature = 0, model_name = "gpt-3.5-turbo")

    chain = LLMChain(llm = llm, prompt = summary_prompt_template)

    linkedin_data = scrape_linkedin_profile()

    print(chain.run(information = linkedin_data))




