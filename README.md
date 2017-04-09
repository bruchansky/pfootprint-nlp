# Political Footprints: Analysing Political Discourses Using Pre-Trained Word Vectors

A political footprint is vector-based representation of a political discourse in which each vector represents a word. Political footprints are computed using machine learning technologies, which allows for systematic and less biased political analysis. 

This toolkit provides heuristics on how to use pre-trained word vectors to analyze public declarations and political debates. Resulting data is unbiased in the sense that they are not relying on the researcher or journalist's political knowledge and beliefs. It is however strongly dependent on the corpus it was trained with (i.e. Wikipedia, Google News), and more generally on the cultural context any political discourse is originating from.

The scripts were designed with journalists, political and cultural researchers in mind. Only a personal computer and little coding knowledge (python) are required to make your own analysis. Assuming your data is well formatted, you can analyze a full presidential election in only half a day. This project is an initiative of [Plural](https://plural.world): a think tank dedicated to pluralism in culture, economics, and politics. Here are a couple of examples of political footprints in action:
- Looking Back at the U.S. Election 2016: A President, but for what Country? (coming soon)
- Is the American Project Social or Economic? (coming soon)

Feel free to contribute and let us know about your projects. The official page of the project will be updated with new contributors and researches: http://plural.world/research/political-footprints. And please reference it in your analysis. 

Finally, here are two papers for those interested to dig deeper in this family of techniques, getting into details of the assumptions being made, their pros and cons, and possible improvements.
- Analysing Political Discourses Using Pre-Trained Word Vectors (computer/political sciences, coming soon)
- Machine Learning: A Structuralist Discipline (philosophy, coming soon)


In the words of Claude Lévi-Strauss:
> « Quant aux créations de l'esprit humain, leur sens n'existe que par rapport à lui, et elles se confondent au désordre dès qu'il aura disparu »


## What is a political footprint?

A political footprint is vector-based representation of a political discourse in which each vector represents a word, with optionally some properties attached, such as a relevance, sentiment, and emotion. It is a subclass of [word vector models](https://www.tensorflow.org/tutorials/word2vec) (or word embeddings) which have the property of locating words with similar meaning closer to one another. Political footprints provide thus a way to “map” semantically a political discourse.
Political footprints are focussing exclusively on what a statement or speaker says. They are thus very different from popular social media word cloud analsysis that focus typically on twitter or news trends. The emphasis is on what a speaker has in his or her control.

![political footprints](https://github.com/Plural-thinktank/pfootprint/blob/master/images/footprints.png)
*From left to right: Sanders, Clinton, and Trump political footprints during U.S. election debates, with words similar to "people" highlighted*.

The current implementation political footprints is based on the following technologies:
- [IBM Watson natural language understanding](https://www.ibm.com/watson/developercloud/natural-language-understanding.html): returns a list of keywords and entities included in a text, with for each a relevance, sentiment, and emotion score.
- [GloVe](https://nlp.stanford.edu/projects/glove/) (Stanford University): english word vectors trained using Wikipedia (2014) and Gigaword 5 (2011). Using this model means that our results will be based on a 2014 snapshot of how words relate to one another. Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation. [pdf](https://nlp.stanford.edu/pubs/glove.pdf) [bib](https://nlp.stanford.edu/pubs/glove.bib).
- [Fasttext](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md) (Facebook): alternative to GloVe useful for non-English texts. P. Bojanowski*, E. Grave*, A. Joulin, T. Mikolov, [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606).
- [Tensorflow and Tensorboard](https://www.tensorflow.org/): Open-source software library for Machine Intelligence, particularly useful in our case for word embeddings visualization.
- [Wordle](http://www.wordle.net/): free online tool to visualize cloud clouds.

![NLP tools](https://github.com/Plural-thinktank/pfootprint/blob/master/images/NLP-tools.png =200x)

This toolkit is multi-language, please consult IBM Watson and Fasttext documentation to see what languages are supported. 

Examples below are based on 2016 U.S. elections. A transcript of all televised debates are available on the [The American Presidency Project](http://www.presidency.ucsb.edu/debates.php).

## Installation
Scripts are written in python and use primarly the tensorflow library. Tensorflow provides some good documentation on how to install both: [install python and tensorflow](https://www.tensorflow.org/install/).

Once both are installed (using Virtualenv on Mac for instance), running the scripts will prompt you with an error message if any library is missing, which can be easily fixed using the sudo command. 

In addition, you need to download and unzip the pretrained space vector model of your choice, for instance [glove.6B.zip](https://nlp.stanford.edu/projects/glove/) or [fasttext](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md) and put the files in a folder of your choice.

## Text format and tokenisation
The first step is gather your data, clean it and format it so that it can be vectorised using tensorflow. Examples of raw data is provided in the US-Elections-2016/debates directory (source: [The American Presidency Project](http://www.presidency.ucsb.edu/debates.php)).  Two scripts have been developed to help you in this process: 

### Step 1: parse html files using pfootprint-generate-texts.py (optional)
Use this script if your data is in the form of html files, or if you deal with a corpus involving more than one contributor per data file. 
```
python pfootprint-generate-texts.py -d US-elections-2016/Primary_Republicans/ -t debate
```

Arguments:
- d: directory with all your html files.
- t: type, either "debate" or "declaration".
In case of a declaration, each html file generates one text file. In case of a debate, the script splits html files into
individual contributions (assuming each contribution starts with the name of the contributor in bold), and creates one text file per contributor.

![step1](https://github.com/Plural-thinktank/pfootprint/blob/master/images/step1.png)

In case you are analyzing a debate, you need to manually clean the text files after running the script:
- Merge files that might have resulted from a slightly different spelling of participant names.
- Delete moderator files. We lose some important information in doing so, making it sometimes difficult to understand what answers from a candidate were about, but it’s the price to pay to take into account only candidates' own words.
- The script names text files using their original folder name (i.e. "Primary_Democrats-SANDERS.txt"), so that we can group political footprints into subcategories, such as ‘Primary-Republicans’, ‘Primary-Democrats’, and ‘General-Election’. If you decide to do so, remember to copy all your text files into a single folder at the end of the process ("text-files" in our example).
Note: The script removes any text within brakets such as "applause", "laugh", etc. We want to focus on what a speaker says and exlude any reaction from the public.

### Step2: parse text files using pfootprint-generate-jsons.py
The second step is to identify key terms in each text file, their relevance, sentiment and emotion attached. The script is using [IBM Watson natural language understanding](https://www.ibm.com/watson/developercloud/natural-language-understanding.html) in order to do so (30 days free trial available). Resulting json files are all saved in a separate folder "json-footprints". Please update your IBM Watson USERNAME and PASSWORD in the script before using it.
```
python pfootprint-generate-jsons.py -d US-Elections-2016/
```

Arguments:
- d: project directory
- l: language (default is 'en'), see IBM docs for supported languages
- u: url (optional)
The script finds all .txt files in a directory (and its subdirectories) and create a "json-footprints" folder with .json files generated by IBM Watson. IBM Watson truncates queries with text files larger than 50kb, a workaround is to upload your files on a server and to use their url instead (limit is 600kb).

![step2](https://github.com/Plural-thinktank/pfootprint/blob/master/images/step2.png)

Notes:
- Using IBM Watson is convenient but it comes with a cost. There is not much control or understanding of how the terms are selected and ponderated. A possible improvement would be to use some open source libraries.
- Two json files are created per text: one for its entities, and the other for its keywords. It’s not clear how IBM Watson creates the two lists, and it is assumed that entities are more relevant than keywords (if a term exists in both lists, we only keep the entity version).

It is at the stage already possible to make some analsysis (fairly well established method). Here is for instance a comparison of key terms used in the Kyoto protocole and Paris agreeement (source:[United Nations Framework Convention on Climate Change](http://newsroom.unfccc.int/)). Word clouds have been rendered using [Wordle](http://www.wordle.net/).

![Kyotot Protocol](https://github.com/Plural-thinktank/pfootprint/blob/master/images/kyoto-protocol-climate-change.png)
*Key terms used in Kyoto Protocol.*

![Paris agreement](https://github.com/Plural-thinktank/pfootprint/blob/master/images/Paris-agreement-climate-change.png)
*Key terms used in Paris agreement.*

Not surprisingly, the texts are mostly neutral, and perhaps a slightly more positive in the Paris agreement. The former puts en emphasis on change and intergovernemtal actions, and the later on economics and sustainibility.

## Create political footprints using tensorflow
At this stage, the only data that matters are the json files we have stored in our JSON-footprint directory. We are now going to leverage this information using pre-trained word vectors by calling the following script:
```
python pfootprint.py -d US-Elections-2016/ -p pretrained-models/glove.6B/glove.6B.300d.txt
```

Arguments:
- d: project directory
- p: pretrained word vector model file (i.e. glove.6B.300d.txt)
The script finds all .json files files in a directory (and its subdirectories) and creates, based on them, a model folder with all political footprints. GloVe has the advantage that we can quickly run the script for a relatively low number of dimensions ("glove.6B.50d.txt"), and increase it to 300 once you are satisfied with the results.

This is it! We now have all our data available in the model folder and we can call tensorboard to see the results:

```
tensorboard --logdir=US-Elections-2016/model
```

Open your browser, go to localhost:6006, select ‘embeddings’ and you should see the following screen with all your results.

![tensorboard](https://github.com/Plural-thinktank/pfootprint/blob/master/images/tensorflow.png)

You should play a bit with it to gain an intuitive understanding of the data. Here is how to use tensorboard: 
- Tensors (left menu): list of all political footprints. In the case of debates, you can also visualize all words that have been used (Aggregation tensor) and compare the location of political footprints centroids (sort of centers of gravity, see heuristic #3). 
- Color by: this option is useful to get a quick and intuitive understanding of political footprints in terms of relevance, sentiments, and emotions. Square sprites have been added to the graph in order to check word sentiments even when some words are preselected (red is negative and blue positive).
- If you wish to visualize the graph in 2-D, run the t-sne for 5000 iterations instead of PCA (bottom left). Keep in mind however that both visualizations are a dimension reduction (the real space has up to hundreds of dimensions!).
- The most useful tool is the list that you see on the right when you select a word: they are the closest words in the original dimension space along with their accurate “distance” (cosine similarity)

So what do we see? Words seem to be grouped in fairly mundane categories: countries tend to be located next to one another, same for politician names, business vocabulary, etc. 

What’s important to understand, and what is really the whole point of using this technique, is that no human has defined these clusters, they have been discovered by machines based on how frequently words occur together, either on Wikipedia or other large corpus of text. This is what allows us to make an unbiased analysis of political discourses. To be clear, there is a strong cultural biais: the one coming from how these words are used on Wikipedia, news feeds, and other large corpora. But it is not coming from the journalist or researcher's point of view.

Let’s now look at how we can practically use political footprints and avoid as much as possible personal interpretations.

## Political footprint analysis
### Heuristic #1: main themes of a discourse and what they mean
1. Open a discourse or candidate tsv file (model folder), using Google Sheets for instance. 
2. Sort the words by relevance and pick up the top 5
3. For each of these words, look inside tensorboard what are the 10 closest words.
4. Select these words in the tsv file and use a tool such as [Wordle](http://www.wordle.net/) to visualise them, with their size = their relevance, and their color = emotion expressed during their occurances.
* Undetected: #a8a8a8
* Joy: #7afff7
* Anger: #b33939
* Disgust: #8f852c
* Fear: #b55c2a
* Sadness: #999370

![Hillary Clinton - Affordable Care Act](https://github.com/Plural-thinktank/pfootprint/blob/master/images/2016-clinton-footprint-care.png)
*Hillary Clinton’ topics that were related to the affordable care act (U.S. election televised debates). Read the full analysis.*

Notes:
- The same heuristic has been applied to both 2008 and 2016 US elections with results that were surprisingly consistent with our intuition. Emotion detection wasn't as reliable. The following Google Spreadsheet formula has been applied in an attempt to reduce dissonances but emotions are still a hit and miss:
```
```

The first word cloud is using only the information returned by IBM Watson.  As you can see, they most often succeed at isolating the most relevant themes for a text. Emotions are a bit less reliable and the equation above attemps to reduce noise. It is important to keep in mind that an emotion used for a word does not represent the feeling TOWARD this word. For instance, an angry feeling to describe people doesn’t mean that the discourse is showing some anger TO people, but that people are desribed in an anger context.
The second word cloud is using words proximity and must be understood with care. Words proximity doesn’t belong to one discourse, but to a culture. What is significant however is the choice of these words in a discourse, their relevance and emotion attached. This is a structuralist approach where a culture and vocabulary is assumed static, and a discourse is picking some words to describe its stance. Comparing distances between words doesn’t give any information about a specific discourse, but only about the culture and language it is based on. Information lies in the choice of these words instead of others, their relevance and emotion associated per discourse. So, in our example, the proper way to describe this cloud is to say that Hillary Clinton has chosen these words to describe healthcare, to highlight … and attach some emotion.
The difference between this word cloud and very similar others that you can see everywhere is that it has not been curated manually: it has been produced automatically based on word distances and then represents a relative objectivity. The researched is not imposing there themes, they are given by the script based on the heuristic described above.

See full example

Climate change: NOT human (except removing change)
Intergovernemental smaller, choice of vocabulary different, 

### Heuristic #2: compare how a theme is interpreted between different discourses
Open the tensorboard aggregated tensor and explore all the concepts and keywords used in a corpus (during an election)
Choose one related to your study. Good candidates are relatively general and ambigious words that can be used in very different ways in each discourse (e.g. “ethical”, “principle”, “values”, “freedom” that can mean very different things)
For that word, select in tensorboard the 20 closest words and in how many discourses each one has been used. Choose the ones the most relevant that you wish to analyze.
Compare using the same steps as the heuristic how each discourse is using them.
This heuristic is useful to identify real paradigm differences between discourses or candidates. The power in this heuristic is that it can compare any text and they don’t necessary have to use the same terms. Because we use a pre-trained word space, we can define the closest words between words being used in different contexts, without any explicit link in each. 

![McCain social matters](https://github.com/Plural-thinktank/pfootprint/blob/master/images/Mccain-social.png)
*John McCain’s topics that were related to social matters (2008 primaries debates).*

![Sanders social matters](https://github.com/Plural-thinktank/pfootprint/blob/master/images/Sanders-social.png)
*Bernie Sanders’ topics that were related to social matters (2016 primaries debates).*

### Heurisitc #3: identify discourse styles and affinity
Open each discourse using tensorboard. Visualze them based on relevance, sentiment, anger, etc. and spot major differences.
Go to the centroids section and compare how close each discourses are.
This heuristic has been the less conclusive based on the current implementation. Some rough analysis can be done but are subject to criticism. For instance, looking at the relevance of the Sanders political footprint highlights the fact is has been very much focussed to wall street, in comparison to Trump or Hillary who have used a more diverse vcabulary. This is not working all the time though.
Clinton emotions not detectable by machine learning, focusses on jobs and affordable care, broad campaign
Centroids are at the stage not very reliable either, probably because they don’t incorporate relevance sentiments and emotions (A discourse seeing positively Islam as a minor theme would be seen equivalent to one is which Islam is central and perceived negatively). We can notice however that for instance Sanders footprint is the furthest away from Trump’s footprint, but it might just be a coincidence. 
## Conclusions
Even in its rudimentary form, political footprints 
Word vector representations are the 21st century version of dictionnaries, or probably even more important for human civilization. Since they will be used by technology to guide our way of thinking. It is very tempting to consider them a reflection of reality, and to take the structuralist approach, but they are only way to understand things. 
