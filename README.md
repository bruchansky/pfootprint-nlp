# Political Footprints: Analysing Political Discourses Using Pre-Trained Word Vectors

A political footprint is vector-based representation of a political discourse in which each vector represents a word. Political footprints are computed using machine learning technologies, which allows for systematic and more objective political analysis. 

This toolkit provides heuristics on how to use pre-trained word vectors to analyse public declarations and political debates. Political footprints are unbiased in the sense that they are not relying on the researcher or journalist's political knowledge or beliefs. They are however very much dependent on the corpus they were trained with (Wikipedia, Google News, etc.), and more generally on the cultural context any political discourse is originating from.

The scripts were designed with journalists and researchers in mind (cultural studies, political science). Only a personal computer and basic knowledge in coding (python) are required to make your own analysis. Assuming your data is well formatted, it is possible to analyse a full presidential election in a single day. This project is an initiative of [Plural](https://plural.world): a think tank dedicated to pluralism, and more specifically the coexistence of multiple cultural, economical, and political paradigms within a same society. Here are some examples of political footprints in action: 
- [Looking Back at the U.S. Election 2016: A President, but for what Country?](https://medium.com/plural-world/looking-back-at-the-u-s-election-2016-a-president-but-for-what-country-949fcc1d8ad7)
- Is the American Project Social or Economic? (coming soon)

Please feel free to contribute to the project. Its official page is on the Plural think tank website, and will be updated with new contributions and applications: http://plural.world/research/political-footprints (link to be used in all references).

Here are two papers for anyone interested to read a bit more about this family of techniques, their assumptions, pros and cons, possible improvements and implications.
- Analysing Political Discourses Using Pre-Trained Word Vectors (computer/political sciences, coming soon)
- Machine Learning: A Structuralist Discipline (philosophy, coming soon)

In the words of Claude Lévi-Strauss:
> « Quant aux créations de l'esprit humain, leur sens n'existe que par rapport à lui, et elles se confondent au désordre dès qu'il aura disparu »

## What is a political footprint?

A political footprint is vector-based representation of a political discourse in which each vector represents a word, with optionally some properties attached, such as a relevance, sentiment, and emotion. It is a subclass of [word vector models](https://www.tensorflow.org/tutorials/word2vec) (or word embeddings) in which words with similar meanings are located close to one another.
Political footprints are sorts of semantic maps applied to political discourse, they have been inspired by other initiatives such as [word2vec4everything](https://github.com/nchah/word2vec4everything). They focus exclusively on what a statement or speaker says. They are, in this sense, very different from other popular word cloud analysis that focus on news articles and social media (i.e. tweets). The emphasis is on what a speaker has in his or her control.

![political footprints](https://github.com/Plural-thinktank/pfootprint/blob/master/images/footprints.png)
*From left to right: Bernie Sanders, Hillary Clinton, and Donald Trump’s political footprints during 2016 U.S. election debates, with the closest words to “people” highlighted.*

The current implementation is based on the following technologies:
- [IBM Watson natural language understanding](https://www.ibm.com/watson/developercloud/natural-language-understanding.html): returns a list of entities and keywords included in a text, with for each a relevance, sentiment, and emotion score.
- [GloVe](https://nlp.stanford.edu/projects/glove/) (Stanford University): package including word vectors trained using Wikipedia (2014) and Gigaword 5 (2011). Using this model means that our results are based on a 2014 snapshot of how words relate to one another. Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation. [pdf](https://nlp.stanford.edu/pubs/glove.pdf) [bib](https://nlp.stanford.edu/pubs/glove.bib).
- [FastText](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md) (Facebook): alternative to GloVe that supports many non-English languages. P. Bojanowski*, E. Grave*, A. Joulin, T. Mikolov, [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606).
- [TensorFlow and TensorBoard](https://www.tensorflow.org/): Open-source software library for machine intelligence, its TensorBoard interface is particularly useful for quick word embeddings visualisation.
- [Wordle](http://www.wordle.net/): free online tool to render word clouds.

![NLP tools](https://github.com/Plural-thinktank/pfootprint/blob/master/images/NLP-tools.png)

This toolkit is multi-language, please consult IBM Watson and FastText documentation to see what languages are supported. 

Examples below are based on 2016 U.S. elections. Transcripts of all televised debates are available on the [American Presidency Project](http://www.presidency.ucsb.edu/debates.php).

## Installation
Scripts are written in python and use, among others, TensorFlow and IBM Watson libraries. TensorFlow provides some good documentation on how to install python and its library on your computer: [install python and TensorFlow](https://www.tensorflow.org/install/).

Once both are installed (using Virtualenv on Mac for instance), running the scripts will prompt you with an error message if any library is missing, which can be easily fixed using the “sudo” command. 

In addition, you need to download and unzip the pre-trained word vector model of your choice, for instance [glove.6B.zip](https://nlp.stanford.edu/projects/glove/) or [FastText](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md) and put the files in your project’s folder.

## Text format and tokenisation
The first step is gather your data, clean it and format it so that it can be vectorised using TensorFlow. Examples of raw data is provided in the examples/US-Elections-2016/debates directory (source: [The American Presidency Project](http://www.presidency.ucsb.edu/debates.php)). Two scripts have been developed to help you in this process: 

### Step 1: parse html files using pfootprint-generate-texts.py (optional)
Use this script if your data is in the form of html files, or if you deal with a corpus involving more than one contributor per data file (debate transcripts). 
```
python pfootprint-generate-texts.py -d examples/US-elections-2016/Primary_Republicans/ -t debate
```

Arguments:
- d: directory with all your html files.
- t: type, either "debate" or "declaration".
In case of declarations, each html file generates one text file. In case of a debate, the script splits html files into individual contributions (assuming each contribution starts with the name of the contributor in bold), and creates one text file per contributor.

![step1](https://github.com/Plural-thinktank/pfootprint/blob/master/images/step1.png)

If you are analysing a debate, you need to manually clean the text files after running the script:
- Merge files that might have resulted from a slightly different spelling of participant names.
- Delete moderator files. We lose some important information in doing so, making it sometimes difficult to understand what answers from a candidate were about, but it’s the price to pay to take into account only candidates' own words.
- The script names text files using their original folder name (i.e. "Primary_Democrats-SANDERS.txt"), so that we can group political footprints into subcategories, such as ‘Primary-Republicans’, ‘Primary-Democrats’, and ‘General-Election’. If you decide to do so, remember to copy all your text files into a single folder at the end of the process ("text-files" folder in our example).

Note: The script removes any text within brackets such as "applause", "laugh", etc. We want to focus on what a speaker says and to exclude any reaction from the public.

### Step2: parse text files using pfootprint-generate-jsons.py
The second step is to identify key terms in each text file, their relevance, sentiment and emotions attached. The script is using [IBM Watson natural language understanding](https://www.ibm.com/watson/developercloud/natural-language-understanding.html) in order to do so (30 days free trial available). Resulting json files are all saved in a separate folder "json-footprints". Please update your IBM Watson USERNAME and PASSWORD in the script before using it.
```
python pfootprint-generate-jsons.py -d examples/US-Elections-2016/
```

Arguments:
- d: project directory
- l: language (default is 'en'), see IBM docs for supported languages
- u: url (optional)
The script finds all .txt files in a directory (and its subdirectories) and create a "json-footprints" folder with .json files generated by IBM Watson. IBM Watson truncates queries with text files larger than 50kb, a workaround is to upload your files on a server and to use their url instead (limit is 600kb).

![step2](https://github.com/Plural-thinktank/pfootprint/blob/master/images/step2.png)

Notes:
- Using IBM Watson is convenient but it comes with a cost: there is not much control or explanation on how the terms are selected and weighted. A possible improvement would be to use some open source libraries.
- Two json files are created per text: one for its entities and one for its keywords. It’s not clear how IBM Watson creates the two lists, and it is assumed that entities are more relevant than keywords (if a term exists in both lists, only the entity version is kept).

It is at the stage already possible to make some fairly well established data analysis. Here is for instance a comparison of key terms used in the the Kyoto protocol and the Paris agreement (source:[United Nations Framework Convention on Climate Change](http://newsroom.unfccc.int/)). Word clouds have been rendered using [Wordle](http://www.wordle.net/).

![Kyotot Protocol](https://github.com/Plural-thinktank/pfootprint/blob/master/images/kyoto-protocol-climate-change.png)

*Key terms used in Kyoto Protocol.*

![Paris agreement](https://github.com/Plural-thinktank/pfootprint/blob/master/images/Paris-agreement-climate-change.png)

*Key terms used in Paris agreement.*

Not surprisingly, the two texts have a neutral sentiment, with perhaps a slightly more positive tone in the Paris agreement. The former puts en emphasis on change and intergovernmental actions, and the later on economics and sustainability.

## Create political footprints using TensorFlow
At this stage, the only data that matters are the json files we have stored in our JSON-footprint directory. We are now going to leverage this information using pre-trained word vectors:
```
python pfootprint.py -d examples/US-Elections-2016/ -p pretrained-models/glove.6B/glove.6B.300d.txt
```

Arguments:
- d: project directory
- p: pre-trained word vector model file (i.e. glove.6B.300d.txt)
The script finds all .json files files in a directory (and its subdirectories) and creates a model folder with all political footprints. Using GloVe has the advantage that we can quickly run the script for a relatively low number of dimensions ("glove.6B.50d.txt"), and increase it to 300 once you are satisfied with the results.

This is it! We now have all processed all our data and we can use TensorBoard to visualise our results:

```
tensorboard --logdir=examples/US-Elections-2016/model
```

Open your browser, go to localhost:6006, select ‘embeddings’ and you should see the following screen with all your results.

![tensorboard](https://github.com/Plural-thinktank/pfootprint/blob/master/images/tensorflow.png)

Play a bit with TensorBoard to gain an intuitive understanding of your data. Here are few tips on how to use TensorBoard: 
- Tensors (left menu): list of all political footprints. In the case of debates, you can also visualise all words that have been used during the debates (Aggregation tensor) and compare the location of political footprints centroids (sort of centres of gravity, see heuristic #3). 
- Colour by: this option is useful to get a quick understanding of political footprints in terms of relevance, sentiment, and emotions. Square sprites have been added to the graph in order to check word sentiments even when some words are selected (red is negative and blue is positive).
- If you wish to visualise the graph in 2-D, run the t-sne for 5000 iterations instead of PCA (bottom left). Keep in mind however that both visualisations are a dimension reduction (the real space has up to hundreds of dimensions!).
- The most useful tool is the list that appears on the right when you select a word: they are its closest words in the original dimension space, based on their cosine similarity (sort of distance between them).

So what do we see? Words seem, at least in the reductive representation of the original space, to be grouped in fairly mundane categories: countries tend to be located next to one another, same for politician names, business vocabulary, etc. 

What’s important to understand, and what is really the whole point of using this technique, is that these clusters have been defined without any human intervention: they have been discovered by machines based on how frequently words appear together, either on Wikipedia or other large corpus of text. This is what allows us to make an unbiased analysis of political discourses. To be clear, there is a strong cultural bias: the one coming from how these words have been used on Wikipedia, news feeds, and other large corpora. But it is not coming from the individual performing the analysis.

Let’s now look at how we can practically use political footprints, and avoid as much as possible using our personal interpretation.

## Political footprint analysis
### Heuristic #1: main themes of a discourse and what they mean
1. Open a discourse or candidate tsv file (model folder), using Google Sheets for instance. 
2. Sort the words by relevance and pick up the top 5.
3. For each of these words, look inside TensorBoard what are the 10 closest words.
4. Select these words in the tsv file and use a tool such as [Wordle](http://www.wordle.net/) to visualise them, with their size = their relevance, and their colour = the emotion they have been associated with.
```
Undetected: #a8a8a8
Joy: #7afff7
Anger: #b33939
Disgust: #8f852c
Fear: #b55c2a
Sadness: #999370
```

![Hillary Clinton - Affordable Care Act](https://github.com/Plural-thinktank/pfootprint/blob/master/images/2016-clinton-footprint-care.png)

*Hillary Clinton’ topics that were related to the affordable care act (U.S. election televised debates). Read the full analysis [here](https://medium.com/plural-world/looking-back-at-the-u-s-election-2016-a-president-but-for-what-country-949fcc1d8ad7).*

Notes:
- The same heuristic was applied to both 2008 and 2016 US elections and results were surprisingly consistent with our intuition.
- Emotion detection wasn't as reliable. The following Google Spreadsheet formula was applied to tsv files in order to reduce noise, the formula only keeps emotions that are consistent with sentiments (joy is only kept when sentiment is positive) and have big enough score. A much better solution would be to use an emotion detection mechanism that can adapt to political language, such as this [this one](https://github.com/lrheault/emotion)(Rheault, Ludovic, Kaspar Beelen, Christopher Cochrane and Graeme Hirst, see their [paper](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0168843)). In any case, it is important to keep in mind that an emotion attached to a word is not necessary targeting that word. An angry feeling detected when using the word people doesn’t mean that the discourse is necessary expressing anger TOWARDS people, but that speaking about people generates some anger. Sarcasm is another example of why emotions detected by IBM Watson might fail. 

```
if(AND(F2>0.1,J2>G2,J2>H2,J2>I2,J2>K2),"Joy",if(AND(F2<-0.1,G2>0.4,G2>H2,G2>H2,G2>J2),"Anger",if(AND(F2<-0.1,H2>0.5,H2>G2,H2>I2,H2>J2),"Disgust",if(AND(F2<-0.1,I2>0.4,I2>G2,I2>H2,I2>J2),"Fear",if(AND(F2<-0.1,K2>0.4),"Sadness","Undetected")))))
with the F column for sentiment, G for anger, H for disgust, I for fear, J for joy, and K for sadness.
```
- Machine learning techniques such as this one belong to structuralism : word similarities are not inferred from the discourse we analyse but from the large corpus of text that was used to train our words (i.e. Wikipedia). Comparing distances (cosine similarity) between words doesn't provide any information about a specific discourse, but only about the culture and language it is based on; information about the discourse lies in the choice of these words instead of others, their relevance and emotion attached. Read this paper for a more detailed analysis  - coming soon.
- A possible extension is to automate this heuristic using k-means clustering or other unsupervised machine learning techniques.

### Heuristic #2: compare how a theme is appropriated by each participant
1. Open the TensorBoard "Aggregated" tensor and choose a term or theme that you would like to study. Let's say we are interested in American "values".
2. For this term, select in the aggregated tensor the 20 closest ones (i.e. "social", "civilization", "inequality", "liberty", etc.) and see how many participants have used them. Choose a couple of terms, ideally those that have been used by many participants and that have many different meanings. "Social" and "interest" were picked in our 2016 US election example, but it could also have been “ethical”, “principle”, “values”, “freedom”: any term that can be used with different sets of words depending on a participant's political views.
3. Compare related terms for each candidate using the same steps as in heuristic #1.

![McCain social matters](https://github.com/Plural-thinktank/pfootprint/blob/master/images/Mccain-social.png)

*John McCain’s topics that were related to social matters (2008 primaries debates). See full example here (coming soon).*

![Sanders social matters](https://github.com/Plural-thinktank/pfootprint/blob/master/images/Sanders-social.png)

*Bernie Sanders’ topics that were related to social matters (2016 primaries debates). See full example here (coming soon).*

This heuristic worked great to identify fundamental differences in how US presidential candidates understood a topic or addressed a problem. One of its benefits is that the relation between words have been defined at the GloVe level, we can thus compare texts that haven't necessary used the same terms to address a same issue. 

### Heuristic #3: sort discourses by style and affinity
1. Open each discourse using TensorBoard. Visualise them coloured by relevance, sentiment, anger, joy, etc. 
2. Open the TensorBoard "Centroids" tensor and compare how far centroids are from one another.

This heuristic has not been conclusive in the current implementation. A couple of interesting properties were revealed in our US election example, but not enough to exclude the possibility that they were mere coincidences:
- Bernie Sanders's political footprint looked more focussed than the others: Wall Street was detected as being by far the most relevant of his topics. But this might have been due to a lack of synonyms to describe a same situation. We can’t conclude that Bernie Sanders was fundamentally more focussed.
- Hillary Clinton's emotions were less visible in her political footprint. But as explained above, this might have been due to her use of sarcasms, and more subtle ways to express her emotions. 
- Except maybe the fact that Bernie Sanders' centroid (centre of gravity) was the furthest from Donald Trump’s, centroids were not corresponding to any of our intentions. Using centroids is arguably a simplistic way to look at political footprints. Centroids are not taking into account relevance, sentiments and emotions: seeing Islam as a positive and secondary topic counts exactly the same as seeing Islam as a negative and central topic.
