# Political Footprints: Political Discourse Analysis Using Pre-Trained Word Vectors

A political footprint is a vector-based representation of a political discourse in which each vector represents a word. Political footprints are computed using machine learning technologies, which allows for a systematic and more objective political analysis. 

This toolkit provides heuristics on how to use pre-trained word vectors to analyze public declarations and political debates. Because political footprints compute semantic similarity based on large corpora of text, they lead to political discourse analysis that relies less on the researcher’s or journalist's political knowledge or beliefs. They are, however, very much dependent on the corpus they were trained with (Wikipedia, Google News, etc.), and, more generally, on the cultural context all political discourse originates in.

The scripts were designed with journalists and researchers in mind (political science and cultural studies). You only require a personal computer and basic knowledge in coding (python) to produce your own analysis. Assuming your data is well formatted, it is possible to analyze a full presidential election in a single day.

This project is an initiative from [Plural](https://plural.world): a think tank dedicated to pluralism, and more specifically the coexistence of multiple cultural, economic, and political paradigms within the same society. In times when soundbites and social media trends are more commented on than any political vision, political footprints aim to help place political discourse back in the center of public debate. Here are some examples of political footprints in action: 
- [Mission-Driven Innovation - From Empty Rhetoric to Meaningful Impact](https://plural.world/research/from-empty-rhetoric-to-meaningful-impact/).
- [Looking Back at the U.S. Election 2016: A President, but for what Country?](https://medium.com/plural-world/looking-back-at-the-u-s-election-2016-a-president-but-for-what-country-949fcc1d8ad7).
- [Is the American Project a Social or Economic One?](https://medium.com/plural-world/is-the-american-project-a-social-or-economic-one-306615f1980f).

In the words of Claude Lévi-Strauss:
> « Quant aux créations de l'esprit humain, leur sens n'existe que par rapport à lui, et elles se confondent au désordre dès qu'il aura disparu »

> “As for the creations of the human mind, their significance only exists in relation to the mind, and they will fall into general chaos as soon as it disappears”

## What is a political footprint?

A political footprint is a vector-based representation of a political discourse in which each vector represents a word, with a number of option properties attached, such as a relevance, sentiment, and emotion. It is a subclass of [word vector models](https://www.tensorflow.org/tutorials/word2vec) (or word embeddings) in which words with similar meanings are located close to one another.
Political footprints are like semantic maps applied to political discourse; they have been inspired by other initiatives such as [word2vec4everything](https://github.com/nchah/word2vec4everything). They focus exclusively on what a statement or speaker says. They are, in this sense, very different from other popular word cloud analyses that focus on news articles and social media (i.e. tweets). The emphasis is on what a speaker has in his or her control.

![political footprints](https://github.com/Plural-thinktank/pfootprint/blob/master/images/footprints.png)
*From left to right: Bernie Sanders, Hillary Clinton, and Donald Trump’s political footprints during 2016 U.S. election debates, with the closest words to “people” highlighted. Transcripts of all televised debates are available on the [American Presidency Project](http://www.presidency.ucsb.edu/debates.php).*

The current implementation is based on the following technologies:
- [IBM Watson natural language understanding](https://www.ibm.com/watson/developercloud/natural-language-understanding.html): returns a list of entities and keywords included in a text, with a relevance, sentiment, and emotion score for each one;
- [GloVe](https://nlp.stanford.edu/projects/glove/) (Stanford University): package including word vectors trained using Wikipedia (2014) and Gigaword 5 (2011). Using this model means that our results are based on a 2014 snapshot of how words relate to one another. Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation. [pdf](https://nlp.stanford.edu/pubs/glove.pdf) [bib](https://nlp.stanford.edu/pubs/glove.bib);
- [FastText](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md) (Facebook): alternative to GloVe that supports non-English languages. P. Bojanowski*, E. Grave*, A. Joulin, T. Mikolov, [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606);
- [TensorFlow and TensorBoard](https://www.tensorflow.org/): open-source software library for machine intelligence, its TensorBoard interface is particularly useful for quick word embeddings visualisation;
- [Wordle](http://www.wordle.net/): free online tool to generate word clouds.

![NLP tools](https://github.com/Plural-thinktank/pfootprint/blob/master/images/NLP-tools.png)

This toolkit is multi-language, please consult IBM Watson and FastText documentation to see what languages are supported. 

## Installation
Scripts are written in python and use, among others, TensorFlow and IBM Watson libraries. TensorFlow provides some good documentation on how to install python and its library on your computer: [install python and TensorFlow](https://www.tensorflow.org/install/).

In addition, you need to download and unzip the pre-trained word vector model of your choice, for instance [glove.6B.zip](https://nlp.stanford.edu/projects/glove/) or [FastText](https://github.com/facebookresearch/fastText/blob/master/pretrained-vectors.md), and put the files in your project’s folder.

## Text format and tokenization
The first step is to gather your data, clean it and format it so that it can be vectorized using TensorFlow. Examples of raw data are provided in the examples/US-Elections-2016/debates folder (source: [The American Presidency Project](http://www.presidency.ucsb.edu/debates.php)). Two scripts have been developed to help you in this process: 

### Step 1: parse html files using pfootprint-generate-texts.py (optional)
Use this script if your data is in the form of html files, or if you deal with a corpus involving more than one contributor per data file (debate transcripts). 
```
python pfootprint-generate-texts.py -d examples/US-elections-2016/Primary_Republicans/ -t debate
```

Arguments:
- d: directory with all your html files.
- t: type, either "debate" or "declaration".
In the case of declarations, each html file generates one text file. In the case of a debate, the script isolates individual contributions (assuming each contribution starts with the name of the contributor in bold), and creates one text file per contributor.

![step1](https://github.com/Plural-thinktank/pfootprint/blob/master/images/step1.png)

In the case of a debate, you need to manually clean the text files after running the script:
- merge files that might have resulted from a slightly different spelling of participant names;
- delete moderator files. We lose a certain amount of important information in doing so, which means it is sometimes difficult to understand what a candidate's answer was about, but this is the price we pay when we only take into account the candidates' own words;
- the script names text files using their original folder name (i.e. "Primary_Democrats-SANDERS.txt"), so that political footprints can be grouped into subcategories, such as ‘Primary-Republicans’, ‘Primary-Democrats’, and ‘General-Election’. All text files need however to be located in a single folder ("text-files" folder in our example).

Note: the script removes any text within brackets such as "applause", "laugh", etc. We want to focus on what a speaker says. We thus need to exclude any reaction from the public.

### Step2: parse text files using pfootprint-generate-jsons.py
The second step is to identify key terms in each text file, their relevance, sentiment and associated emotions. The script uses [IBM Watson natural language understanding](https://www.ibm.com/watson/developercloud/natural-language-understanding.html) in order to do so (30 days free trial available). Resulting json files are all saved in a separate folder "json-footprints". Please update your IBM Watson USERNAME and PASSWORD in the script before using it.
```
python pfootprint-generate-jsons.py -d examples/US-Elections-2016/
```

Arguments:
- d: project directory
- l: language (default is 'en'), see IBM docs for supported languages
- u: url (optional)
The script finds all .txt files in a directory (and its subdirectories) and creates a "json-footprints" folder with .json files generated by IBM Watson. IBM Watson truncates queries with text files larger than 50kb; a workaround is to upload your files on a server and use their url instead (limit is 600kb).

![step2](https://github.com/Plural-thinktank/pfootprint/blob/master/images/step2.png)

Notes:
- using IBM Watson is convenient but it comes with a cost: there is not much control over or explanation of how the terms are selected and weighted. A possible improvement would be to use open source libraries;
- two json files are created per text: one for its entities and one for its keywords. It is not clear how IBM Watson creates the two lists, and it is assumed that entities are more relevant than keywords (if a term exists in both lists, only the entity version is kept).

## Create political footprints using TensorFlow
At this stage, the only data that matter are the json files we have stored in our JSON-footprint directory. We are now going to leverage this information using pre-trained word vectors:
```
python pfootprint.py -d examples/US-Elections-2016/ -p pretrained-models/glove.6B/glove.6B.300d.txt
```

Arguments:
- d: project directory
- p: pre-trained word vector model file (i.e. glove.6B.300d.txt)
The script finds all .json files files in a directory (and its subdirectories) and creates a model folder with all political footprints. Using GloVe has the advantage that we can quickly run the script for a relatively low number of dimensions ("glove.6B.50d.txt"), and increase it to 300 once you are satisfied with the results.

Done! We now have all processed all our data and we can use TensorBoard to visualize our results:

```
tensorboard --logdir=examples/US-Elections-2016/model
```

Open your browser, go to localhost:6006, select ‘embeddings’ and you should see the following screen with all your results.

![tensorboard](https://github.com/Plural-thinktank/pfootprint/blob/master/images/tensorflow.png)

Play a bit with TensorBoard to gain an intuitive understanding of your data. Here are few tips on how to use TensorBoard: 
- Tensors (left menu): list of all political footprints that have been processed. In the case of debates, you can also visualize all words that have been used during the debates (Aggregation tensor) and compare the location of political footprints centroids (akin to centers of gravity, see heuristic #3). 
- Color by: this option is useful to get a quick understanding of political footprints in terms of relevance, sentiment, and emotions. Square sprites have been added to the graph in order to check word sentiments even when some words are selected (red is negative and blue is positive).
- If you wish to visualize the graph in 2-D, run the t-sne for 5000 iterations instead of PCA (bottom left). Keep in mind, however, that both visualizations are reductions (the real space has up to hundreds of dimensions!).
- The most useful tool is the list that appears on the right when you select a word: they are its closest neighbors in the original dimension space, based on their cosine similarity (the distance between them).

So what do we see? Words seem, at least in the 3D space, to be grouped in fairly mundane categories: countries tend to be located next to one another, the same goes for politician names, business vocabulary, etc. 

What is important to understand, and what is really the whole point of using this technique, is that these clusters have been defined without any human intervention: they have been discovered by machines based on how frequently words appear together, either on Wikipedia or in other large corpora of text. This is what allows us to make an unbiased analysis of political discourse. To be clear, there is a strong cultural bias, stemming from how these words have been used on Wikipedia, news feeds, and other large corpora, but not from the researcher performing the analysis.

Let us now look at how we can put political footprints to practical use, and avoid using our personal interpretation as much as possible.

## Political footprint analysis
### Heuristic #1: main themes of a discourse and what they mean
1. Open a discourse or candidate tsv file (model folder), using, for instance, Google Sheets. 
2. Sort the words by relevance and pick up the top 5.
3. For each of these words, look inside TensorBoard at the 10 closest words.
4. Select these words in the tsv file and use a tool such as [Wordle](http://www.wordle.net/) to visualize them, with their size = their relevance, and their color = the emotion they have been associated with.
```
Undetected: #a8a8a8
Joy: #7afff7
Anger: #b33939
Disgust: #8f852c
Fear: #b55c2a
Sadness: #999370
```



![Paris agreement](https://github.com/Plural-thinktank/pfootprint/blob/master/images/Paris-agreement-climate-change.png)

*Climate-related terms used in Paris agreement (source:[United Nations Framework Convention on Climate Change](http://newsroom.unfccc.int/)). Word clouds have been generated using [Wordle](http://www.wordle.net/).*


![Hillary Clinton - Affordable Care Act](https://github.com/Plural-thinktank/pfootprint/blob/master/images/2016-clinton-footprint-care.png)

*Hillary Clinton’s topics that related to the affordable care act (U.S. election televised debates). Transcripts of all televised debates are available on the [American Presidency Project](http://www.presidency.ucsb.edu/debates.php). Read the full analysis [here](https://medium.com/plural-world/looking-back-at-the-u-s-election-2016-a-president-but-for-what-country-949fcc1d8ad7).*

Notes:
- the same heuristic was applied to both the 2008 and 2016 US elections and the results were surprisingly consistent with our intuition;
- emotion detection was not as reliable. A Google Spreadsheet formula was applied to tsv files in order to reduce noise: the formula only keeps emotions that are consistent with sentiments (joy is only kept when sentiment is positive) and have big enough scores. A much better solution would be to use an emotion detection mechanism that can adapt to political language, such as this [this one](https://github.com/lrheault/emotion)(Rheault, Ludovic, Kaspar Beelen, Christopher Cochrane and Graeme Hirst, see their [paper](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0168843)). In any case, it is important to keep in mind that an emotion attached to a word is not necessary targeting that word. An angry feeling detected when using the word "people" doesn’t mean that the discourse is necessary expressing anger TOWARDS people, but that speaking about people generates anger. Sarcasm is another example of why emotions detected by IBM Watson might fail. 

```
if(AND(F2>0.1,J2>G2,J2>H2,J2>I2,J2>K2),"Joy",if(AND(F2<-0.1,G2>0.4,G2>H2,G2>H2,G2>J2),"Anger",if(AND(F2<-0.1,H2>0.5,H2>G2,H2>I2,H2>J2),"Disgust",if(AND(F2<-0.1,I2>0.4,I2>G2,I2>H2,I2>J2),"Fear",if(AND(F2<-0.1,K2>0.4),"Sadness","Undetected")))))
with the F column for sentiment, G for anger, H for disgust, I for fear, J for joy, and K for sadness.
```
- Machine learning techniques such as this one belong to structuralism: word similarities are not inferred from the discourse that is analyzed but from the large corpus of text that was used to train the words (i.e. Wikipedia). Comparing distances (cosine similarity) between words does not provide any information about a specific discourse, but only about the culture and language it is based on; information about the discourse lies in the choice of these words instead of others, their relevance and the associated emotion. Read this [article](https://medium.com/plural-world/looking-back-at-the-u-s-election-2016-a-president-but-for-what-country-949fcc1d8ad7) for a more detailed analysis;
- a possible extension would be to automate this heuristic using k-means clustering or other unsupervised machine learning techniques.

### Heuristic #2: compare how a theme is appropriated by each participant
1. Open the TensorBoard "Aggregated" tensor and choose a term or theme that you would like to study. Let's say we are interested in American "values".
2. For this term, select in the aggregated tensor the 20 closest terms (i.e. "social", "civilization", "inequality", "liberty", etc.) and see how many participants have used each of them. Choose a couple of terms, ideally those that have been used by many participants and that have many different meanings. "Social" and "interest" were picked in our 2016 US election example, but it could also have been “ethical”, “principle”, “values”, “freedom”: any term that can be used with different sets of words depending on a participant's political views.
3. Compare related terms, to "social" and "interest" in our example, for each candidate using the heuristic #1.

![McCain social matters](https://github.com/Plural-thinktank/pfootprint/blob/master/images/Mccain-social.png)

*John McCain’s topics that related to social matters (2008 primaries debates). Transcripts of all televised debates are available on the [American Presidency Project](http://www.presidency.ucsb.edu/debates.php). See full example [here](https://medium.com/plural-world/is-the-american-project-a-social-or-economic-one-306615f1980f).*

![Sanders social matters](https://github.com/Plural-thinktank/pfootprint/blob/master/images/Sanders-social.png)

*Bernie Sanders’ topics that related to social matters (2016 primaries debates). Transcripts of all televised debates are available on the [American Presidency Project](http://www.presidency.ucsb.edu/debates.php). See full example [here](https://medium.com/plural-world/is-the-american-project-a-social-or-economic-one-306615f1980f).*

This heuristic was very effective in identifying fundamental differences in how US presidential candidates understood a topic or addressed a problem. One of its benefits is that the relationship between words has been defined at the GloVe level; we can thus compare texts that have not necessarily used the same terms to address the same issue. 

### Heuristic #3: sort discourses by style and affinity
1. Open each discourse using TensorBoard. Visualize them colored by relevance, sentiment, anger, joy, etc. 
2. Open the TensorBoard "Centroids" tensor and compare how far centroids are from one another.

This heuristic has not been conclusive in the current implementation. A couple of interesting properties were revealed in our US election example, but not enough to exclude the possibility that they were mere coincidences.
- Bernie Sanders's political footprint seemed more focused than the others: Wall Street was detected as being by far the most relevant of his topics. But this might have been due to his lack of synonyms to describe the same situation. We cannot conclude that Bernie Sanders was fundamentally more focused.
- Hillary Clinton's emotions were less visible in her political footprint. But as explained above, this might have been due to her use of sarcasm, and more subtle ways of expressing her emotions. 
- With the possible exception of the fact that Bernie Sanders' centroid (center of gravity) was the farthest from Donald Trump’s, centroids did not correspond to any of our intuitions. Using centroids is arguably a simplistic way to look at political footprints. Centroids do not take into account relevance, sentiments and emotions: seeing Islam as a positive and secondary topic counts exactly the same as seeing Islam as a negative and central topic.
