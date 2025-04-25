find_target_prompt= """You are given multiple pages of a document, along with a question and an answer that were provided for a query about the document.

**Question:** [Question]
**Answer:** [Answer]

Your task is to finding all pages of the document are necessary to answer the query accurately. You need to return the image number which starts from 0.
Return your findings as a list of page numbers in the following format.
If all of the images seem to be necessary, put all page indices in the list.

*However, if there is no required pages in the given images return an empty list.*
**Each image is a page, even if the image is a crop of a bigger image.**
*Make sure to look back your choice and there must be the **given answer** in the selected pages.*

You can only speak json and put all of your thoughts under the thoughts.

Use this JSON schema:
Return: {'thoughts': str, 'output': [int]}
"""

scroll_step_prompt= """ You are given:
- A single-page image of a document
- A question
- The number of pages to be skipped and scrolled in the next step
- Current page number
- The total number of pages in the document
- Notes containing relevant information from other pages

At each step, you receive this prompt repeatedly, enabling you to scroll through the document page by page to gather information and ultimately answer the question. Imagine yourself as a human analyzing the document, making observations, and reasoning about whether to continue scrolling. Your task is to generate realistic, human-like reasoning for decision-making. Think as if you have the choice to either answer or continue exploring based on your notes and findings, while also determining the appropriate scroll value—though you are not allowed to answer at this step. The first page of the document is page number 0. 
The page numbers in the Previous Note and Current_page_num simply indicates nth image of document screenshots. Thus, if there is a page number stated in the image or prompt, it may be not aligned with the page numbers in the notes and Current_page_num.

**Question:** [Question]
**Previous Note:** [Previous_Note]
**Scroll_value:** [Scroll_value]
**Current_page_num:** [Current_page_num]
**Total_page_num:** [Total_page_num]

Your job is to:
1. Identify any information on the **current page** that can be useful to answer the question.  Do not repeat the previous note and its information. Only return the new information. Also, note a brief summary of the current page. Do not format the note at all. Put a string simply.
2. Write out an in-depth thinking process about how you find this relevant information and reasoning to conclude to scroll by the scroll value (The thoughts should not reveal that it is instructed and the scroll value is provided.. Answer as if you are not given the scroll value. You still need to provide profound reasoning that you need to scroll [Scroll_value]).
Return your response in **JSON format**.
"""

final_step_prompt= """ You are given:

- A single-page image of a document
- A question
- Current page number
- The total number of pages in the document
- Notes containing relevant information from other pages

At each step, you receive this prompt repeatedly, enabling you to scroll through the document page by page to gather information and ultimately answer the question. Imagine yourself as a human analyzing the document, making observations, and reasoning about whether to continue scrolling. Your task is to generate realistic, human-like reasoning for decision-making. Think as if you have the choice to either answer or continue exploring based on your notes and findings without the answer given, though you must provide the final answer to the question in the end. The first page of the document is page number 0.
The any pages including the first and last page may have the enough infromation to answer the question.
The page numbers in the Previous Note and Current_page_num simply indicates nth image of document screenshots. Thus, if there is a page number stated in the image or prompt, it may be not aligned with the page numbers in the notes and Current_page_num.

**Question:** [Question]
**Answer:** [Answer]
**Previous Note:** [Previous_Note]
**Current_page_num:** [Current_page_num]

Try to examine each step in depth and as if it is a realistic thinking process. **Do not include "answer" as a key in the json.** **Never start your thoughts with "Okay".**

Your task is to:
1. Identify any information on the **current page** that can be useful to answer the question.
2. Write out an in-depth thinking process step by step to identify the answer to the question on the current page. There must be the answer in the current page.
3. Explain why you now have enough information to provide the answer
4. You MUST DERIVE THE FINAL ANSWER based on the previous note and the current page. *You cannot say that the answer is given.*

Return your response in the exact following **JSON format**.
"""

scroll_llava_prompt= """At each step, you will receive this prompt repeatedly, enabling you to scroll through the document page by page to gather information and ultimately answer the question. Imagine yourself as a human analyzing the document, making observations, and reasoning about whether to continue scrolling or answer to the question. The first page of the document is page number 0. 
**Question:** [Question]
**Previous Note:** [Previous_Note]
**Current_page_num:** [Current_page_num]
**Total_page_num:** [Total_page_num]
Using given infromation above, you can choose to scroll the document to explore other pages or answer the question.

If you chose to scroll, you should return your thoughts, notes to pass question relevant infroamtion to the next step, and scroll values to scroll forward or backward.
Return the thinking process in <think>...</think>, the notes in <note>...</note>, and the scroll value (+n or -n) in <scroll>...</scroll> tags.

If you chose to answer, you should return your thoughts and final answer to the given question.
Return the thinking process in <think>...</think> and the answer in <answer>...</answer> tags.
"""

answer_llava_prompt= """You need to answer to the question based on the given information. The first page of the document is page number 0. 
**Question:** [Question]
**Previous Note:** [Previous_Note]
**Total_page_num:** [Total_page_num]

You should return your thoughts and final answer to the given question.
Return the thinking process in <think>...</think> and the answer in <answer>...</answer> tags.
"""