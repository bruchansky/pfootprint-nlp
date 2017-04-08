# Political Footprints: Analysing Political Discourses Using Pre-Trained Word Vectors

A political footprint is vector-based representation of a political discourse in which each vector represents a word. Political footprints are computed using machine learning technologies, which allows for systematic and less biased political analysis. 

This toolkit provides heuristics on how to use pre-trained word vectors to analyze public declarations and political debates. Resulting data is unbiased in the sense that they are not relying on the researcher or journalist's political knowledge and beliefs. It is however strongly dependent on the corpus it was trained with (i.e. Wikipedia, Google News), and more generally on the cultural context any political discourse is originating from.

The scripts were designed with journalists, political and cultural researchers in mind. Only a personal computer and little coding knowledge (python) are required to make your own analysis. Assuming your data is well formatted, you can analyze a full presidential election in only half a day. This project is an initiative of [Plural](https://plural.world): a think tank dedicated to pluralism in culture, economics, and politics. Here are a couple of examples of political footprints in action:
- Looking Back at the U.S. Election 2016: A President, but for what Country? (coming soon)
- Is the American Project Social or Economic? (coming soon)

Feel free to contribute and let us know about your projects. The official page of the project will be updated with new contributors and researches: http://plural.world/political-footprints. And please reference it in your analysis. 

Finally, here are two papers for those interested to dig deeper in this family of techniques, getting into details of the assumptions being made, their pros and cons, and possible improvements.
- Analysing Political Discourses Using Pre-Trained Word Vectors (computer/political sciences, coming soon)
- Machine Learning: A Structuralist Discipline (philosophy, coming soon)


In the words of Claude Lévi-Strauss:
> « Quant aux créations de l'esprit humain, leur sens n'existe que par rapport à lui, et elles se confondent au désordre dès qu'il aura disparu »


## What is a political footprint?

It is a vector based representation of a political discourse in which each vector represents a word, with optionally some properties attached to them, such as a relevance, sentiment, and emotion. The most powerful characteristic of these vector-based representations (or word embeddings) is that they locate words with similar meaning closer to one another. Political footprints provide thus a way to “map” a political standpoint expressed in a given context.
Political footprints are focussing exclusively on what a statement or speaker says. It is thus very different from social media word clouds analsysis, such as for Google trends and twitter. The emphasis is on what a speaker has in his or her control.
 
Image of a political footprint
Senator Sanders political footprint during the Democrats primaries debates - 2016 US election

The current implementation of these political footprints is based on the following technologies:
IBM Watson natural language understanding: returns a list of keywords and entities included in a text, with for each a relevance, sentiment, and emotion score.
University of Stanford’s GloVe (Global Vectors for Word Representation): english word vectors trained using Wikipedia (2014) and Gigaword 5 (2011). Using this representation means that our results will be based on a 2014 snapshot of how words relate to one another. This is an important limitation that is studied in details in the related paper, and has profound philosophical implications.
Fasttext developed by facebook: (Creative Commons Attribution-Share-Alike License 3.0.): alternative to GloVe used for other languages than English, and based on the 2016 Wikipedia Corpus.
Tensorflow and tensorboard: An open-source software library for Machine Intelligence.
Wordle: free online tool to visualize cloud clouds.

This tool is multi-language, please have a look at IBM Watson and Fasttext to see if your language is supported. 

Use cases below are based on The American Presidency Project corpus.

Logos partners, multi language, works on PC, american debate
Installation
Many libraries inclusind watson, tensorflow

Data format and tokenisation
The first step is gather your data, clean it and format it so that it can be vectorised by tensorflow. Two scripts are provided to help you in this process: 
You are provided with two scripts to prepare your data:

Step 1: parse html files using pfootprint-generate-texts.py (optional)
Use this script if your data is in the form of html files, or if you deal with a corpus involving more than one contributor per text file. 
python pfootprint-generate-texts.py -d US-elections-2016/Primary_Republicans/ -t debate

Arguments:
-d: directory with all your html files
-t: type, either ‘debate’ or ‘declaration’. In case of a declaration, each html file produces one text file. In case of a debate, the script splits html files into individual contributions (assuming each contribution starts with the name of the contributor in bold), and creates one text file per contributor.
Image step 1

This script removes any instance in brakets such as applause, … because we want to focus on what a speakers says and not take into account people’s reactions.

For debates, manually clean the text files afterwards:
Delete contributions from moderators. We lose some important data in doing so, making it sometimes difficult to understand what declarations are about, but it’s the price to pay to analyze each candidates own words.
Merge text files with name variations for the same contributors.
Text filenames include the original html folder name so that you can perform this operation for several groups, for instance ‘Primary-Republicans’, ‘Primary-Democrats’, and ‘General-Election’. If you do so, bring at the end of the process all your text files into one single folder, for instance ‘text-files’.
This script removes all 
Step2: parse text files using pfootprint-generate-jsons.py
The second step is to identify key terms in each text file, their importance, sentiment, and emotion attached. This script is using IBM Watson Natural Language Understanding in order to do so (30 days free trial), and create json files in a separate folder ‘json-footprints’. First edit it with your IBM Watson and password then use it as following:
Arguments:
-d: directory with all your text files 
-l: language (default is 'en'), see IBM docs for supported languages
-u: url (optional) 
IBM Watson truncates queries with texts larger than 50kb,
a workaround is to upload your files on a server and to
use their url instead (limit is 600kb).

Image process
Using IBM Watson Natural Language Understanding is convenient because it allows to externalize this complex process to an API. It comes with a cost however that there is not much control, or understanding of how the terms are selected and ponderated. This step could be developed as part of the toolkit but it would require much more machine resources and time, without the garantee of achieving better results (candidate for future improvements). The scripts create two json files per text: one for its entities, and the other for its keywords. It’s not very clear how IBM Watson creates the two lists, the pfootprint.py script assumes however that entities are more relevant than keywords, and only consider keywords that were not in the entities list.
Tests so far have shown results that correspond most of the time to the intuition of what important words were used in a speech (see below), but it’s not perfect.
Example result kyoto
Only joy or no feeling, a bit more negative for the kyoto protocole (greenhouse, carbon dioxide, antropohohenic) 
Compare it to other protocols
Create political footprints with tensorflow
At this stage, the only data that matters is the one stored in our JSON-footprint folder. We are now going to leverage this information using pre-trained word vectors. We do this by calling the following script
Arguments
-d: project directory
-p: pretrained file. GloVe has the advantage that you can quickly run the script for a relatively low word dimension (file, 50 dimensions), and increase it to 300 once you are satisfied with the results.
The script will create a model folder with all the information required to run tensorboard (visualisation tool):
Tensorboard
With:
--logdir the directory where the model generated by pfootprint is

This is it! Go to localhost:6006, select ‘embeddings’ and you should see the following screen with all your results.

Here are some tensorboard features to play with:
Tensors (left menu): list of all the tensors created, one per contributor. The script also created an Aggregation tensor to see all the words used in the corpus, and the Centroids tensor to compare each contribution (see heuristic 3)
Color by: this will allow you to have a quick and intuitive understanding of the political footprints in terms of relevance, sentiments, and emotions. The script has created a -sentiment that is the opposite sentiment for an easier visualisation. The square images you see for each word on the political footprint graph have been created to give an identication of related sentiments (red is negative and blue positive) even when a word is selected.
Run the t-sne for 5000 iterations if you wish to see a 2D representation of the space (can be convenient to compare colours). Keep in mind however that everything you see is an approximation because the real vector space can have up to 300 dimensions! See the next point.
The most useful analysis tool is the list you see on the left when you select a word: they are the closest words in the original dimension (up to 300) along with their accurate “distance” (complicated matter to understand, but cosine is the best one)

So what do we see? Words are located based on fairly mundane semantic: countries tend to be located together, same for candidate names, business vocabulary, etc. 

What’s important to understand, and what is really the value of these pre-trained words, is that no human has defined these clusters, they have been “guessed” by machines based on how frequently words occur together, either on wikipedia or other large corpus: artificial intelligence and deep learning incremental algorithms have been applied to these corpus to discover word associations. This is what allows for a relatively unbiased analysis of the political discourses. To be clear, there is a strong cultural biais: the one coming from how these words are used in wikipedia, news feeds, and that biais is specific to a language, a corpus, and the moment of record. But what we are close to achieve is no biais coming from the journalist or researcher performing this research.

Let’s now look at how we can practically use this information, with the least possible personal interpretation. 
Political footprint analysis
Heuristic #1: main themes of a discourse and how they are articulated
Open the a discourse or candidate tsv file in the model folder, using Google sheets for instance. For instance, 
Sort the words (themes) by relevance and take the top 5
For each of these themes, look inside tensorboard what are the ten closest words for that candidate
Select these words in the tsv file and use a tool such as to visualise their attached relevance and emotion.
PS for emotions, it’s a bit tricky
Here is an example: top 5 themes for Hillary…, what health care means
How to interpret the results:
The first word cloud is using only the information returned by IBM Watson. It is a fairly well established way to analyze a text using natural language processing. As you can see, they most often succeed at isolating the most relevant themes for a text. Emotions are a bit less reliable and the equation above attemps to reduce noise. It is important to keep in mind that an emotion used for a word does not represent the feeling TOWARD this word. For instance, an angry feeling to describe people doesn’t mean that the discourse is showing some anger TO people, but that people are desribed in an anger context.
The second word cloud is using words proximity and must be understood with care. Words proximity doesn’t belong to one discourse, but to a culture. What is significant however is the choice of these words in a discourse, their relevance and emotion attached. This is a structuralist approach where a culture and vocabulary is assumed static, and a discourse is picking some words to describe its stance. Comparing distances between words doesn’t give any information about a specific discourse, but only about the culture and language it is based on. Information lies in the choice of these words instead of others, their relevance and emotion associated per discourse. So, in our example, the proper way to describe this cloud is to say that Hillary Clinton has chosen these words to describe healthcare, to highlight … and attach some emotion.
The difference between this word cloud and very similar others that you can see everywhere is that it has not been curated manually: it has been produced automatically based on word distances and then represents a relative objectivity. The researched is not imposing there themes, they are given by the script based on the heuristic described above.

See full example

Climate change: NOT human (except removing change)
Intergovernemental smaller, choice of vocabulary different, 

Heuristic #2: compare how a theme is interpreted between different discourses
Open the tensorboard aggregated tensor and explore all the concepts and keywords used in a corpus (during an election)
Choose one related to your study. Good candidates are relatively general and ambigious words that can be used in very different ways in each discourse (e.g. “ethical”, “principle”, “values”, “freedom” that can mean very different things)
For that word, select in tensorboard the 20 closest words and in how many discourses each one has been used. Choose the ones the most relevant that you wish to analyze.
Compare using the same steps as the heuristic how each discourse is using them.
This heuristic is useful to identify real paradigm differences between discourses or candidates. The power in this heuristic is that it can compare any text and they don’t necessary have to use the same terms. Because we use a pre-trained word space, we can define the closest words between words being used in different contexts, without any explicit link in each. 

Heurisitc #3: identify discourse styles and affinity
Open each discourse using tensorboard. Visualze them based on relevance, sentiment, anger, etc. and spot major differences.
Go to the centroids section and compare how close each discourses are.
This heuristic has been the less conclusive based on the current implementation. Some rough analysis can be done but are subject to criticism. For instance, looking at the relevance of the Sanders political footprint highlights the fact is has been very much focussed to wall street, in comparison to Trump or Hillary who have used a more diverse vcabulary. This is not working all the time though.
Clinton emotions not detectable by machine learning, focusses on jobs and affordable care, broad campaign
Centroids are at the stage not very reliable either, probably because they don’t incorporate relevance sentiments and emotions (A discourse seeing positively Islam as a minor theme would be seen equivalent to one is which Islam is central and perceived negatively). We can notice however that for instance Sanders footprint is the furthest away from Trump’s footprint, but it might just be a coincidence. 
Conclusions
Even in its rudimentary form, political footprints 
Word vector representations are the 21st century version of dictionnaries, or probably even more important for human civilization. Since they will be used by technology to guide our way of thinking. It is very tempting to consider them a reflection of reality, and to take the structuralist approach, but they are only way to understand things. 
