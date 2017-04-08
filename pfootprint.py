''' SCRIPT TO CREATE POLITICAL FOOTPRINTS BASED ON JSON FILES AND
A PRETRAINED WORD VECTOR MODEL
ARGUMENTS:
-d: project directory
-p: pretrained word vector model file (i.e. glove.6B.300d.txt)
This script finds all .json files files in a directory (and its subdirectories)
and creates, based on them, a model folder with political footprints
(political footprints can be later on visualized with tensorboard).'''

import numpy as np
import sys
import getopt
import math
import json
import re
import os
from PIL import Image
from PIL import ImageDraw
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

# GLOBAL VARIABLES
DIR_MODEL = "model"
SPRITE_UNIT_SIZE = 100
centroids_vocab = []
centroids = []
aggregation = []
DIR = ''
config = projector.ProjectorConfig()
sess = tf.Session()


def get_input(argv):
    # GET PARAMETERS -d for directory
    directory = None
    try:
        opts, args = getopt.getopt(argv, "hd:p:", ["dir=", "pretrained="])
    except getopt.GetoptError:
        print 'pfoot.py -d <dir> -p <pretrained>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'pfoot.py -d <dir> -p <pretrained>'
            sys.exit()
        elif opt in ("-d", "--dir"):
            directory = arg
        elif opt in ("-p", "--pretrained"):
            pretrained = arg
    if directory is None:
        print 'pfoot.py -d <dir> -p <pretrained>'
    else:
        return directory, pretrained


# PARSE JSON AND CREATE LIST OF WORDS WITH RELEVANCE AND EMOTIONS
def parse_json(xml, words):
    entities = []
    if 'entities' in xml:
        entities = xml["entities"]
    if 'keywords' in xml:
        entities = xml["keywords"]
    for entity in entities:
        words_per_line = entity["text"].strip().split(' ')
        for word in words_per_line:
            cleaned_word = word.encode('utf-8').lower()
            match = re.match(r'\b(?:[a-zA-Z]\.){2,}', cleaned_word)
            if not match:
                cleaned_word = cleaned_word.rstrip('.')
            tempItem = {}
            tempItem["relevance"] = float(entity.get("relevance", 0))
            if "sentiment" in entity:
                tempItem["sentiment"] = \
                    float(entity["sentiment"].get("score", 0))
            else:
                tempItem["sentiment"] = 0
            tempItem["negative"] = - tempItem["sentiment"]
            if "emotion" in entity:
                tempItem["anger"] = \
                    float(entity["emotion"].get("anger", 0))
                tempItem["disgust"] = \
                    float(entity["emotion"].get("disgust", 0))
                tempItem["fear"] = float(entity["emotion"].get("fear", 0))
                tempItem["joy"] = float(entity["emotion"].get("joy", 0))
                tempItem["sadness"] = \
                    float(entity["emotion"].get("sadness", 0))
            else:
                tempItem["anger"] = 0
                tempItem["disgust"] = 0
                tempItem["fear"] = 0
                tempItem["joy"] = 0
                tempItem["sadness"] = 0
            tempItem["found"] = False  # to indicate if found in GloVe
            if cleaned_word not in words and \
                    len(cleaned_word) > 2:
                words[cleaned_word] = tempItem
            else:
                # only keeps the words with the highest relevance
                print 'Short or duplicate word: ' + cleaned_word
    return words


# CREATE TENSOR & EMBEDDING BASED ON A LIST OF WORDS
def create_tensor(words, name, pretrained):
    global centroids_vocab
    global centroids
    global aggregation
    # match words with GloVe
    vocab = []
    embd = []
    candidates = []
    file = open(pretrained, 'r')
    try:
        os.makedirs(os.path.join(DIR, DIR_MODEL))
    except Exception:
        pass
    metadata = open(os.path.join(DIR, DIR_MODEL, name + ".tsv"), 'w')
    if name == "Aggregation":
        metadata.write("Word\t")
        for word, item in words.items():
            for candidate in item:
                if candidate not in candidates and not candidate == "found":
                    metadata.write(candidate + "\t")
                    candidates.append(candidate)
        metadata.write("Shared\n")
    else:
        metadata.write("Word\tRelevance\t\
                   Sentiment\t-Sentiment\t\
                   Anger\tDisgust\tFear\tJoy\tSadness\n")
    for line in file.readlines():
        row = line.strip().split(' ')
        for word, item in words.items():
            if row[0] == word:
                vocab.append(word)
                embd.append(row[1:])
                item["found"] = True
                if name == "Aggregation":
                    count = 0
                    metadata.write(word + "\t")
                    for candidate in candidates:
                        if candidate in item:
                            metadata.write(str(item[candidate]) + "\t")
                            count = count + 1
                        else:
                            metadata.write("0\t")

                    metadata.write(str(count) + "\n")
                else:
                    metadata.write(word + "\t" + str(item["relevance"]) +
                                   "\t" + str(item["sentiment"]) +
                                   "\t" + str(item["negative"]) +
                                   "\t" + str(item["anger"]) +
                                   "\t" + str(item["disgust"]) +
                                   "\t" + str(item["fear"]) +
                                   "\t" + str(item["joy"]) +
                                   "\t" + str(item["sadness"]) +
                                   "\n")
                break
    file.close()
    metadata.close()
    vocab_size = len(vocab)
    embedding_dim = len(embd[0])
    embedding = np.asarray(embd)
    for word, item in words.items():
        if item["found"] is False:
            print 'Word not found in GloVe: ' + word
    print 'Words matched with GloVe: ' + str(vocab_size)

    # create tensor
    W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
                    trainable=False, name=name)
    embedding_placeholder = tf.placeholder(tf.float32,
                                           [vocab_size, embedding_dim])
    embedding_init = W.assign(embedding_placeholder)
    sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})
    print 'Tensors for', name, 'created'
    centroid = create_centroid(W)
    print 'Centroid for', name, 'created'

    # connect with labels and images
    embedding = config.embeddings.add()
    embedding.tensor_name = W.name
    embedding.metadata_path = os.path.join(DIR, DIR_MODEL, name + ".tsv")
    if not name == "Aggregation":
        sprite = create_sprite(words, vocab)
        sprite.save(os.path.join(DIR, DIR_MODEL, name + "sprite.gif"), "GIF")
        embedding.sprite.image_path = \
            os.path.join(DIR, DIR_MODEL, name + "sprite.gif")
        embedding.sprite.single_image_dim.extend([SPRITE_UNIT_SIZE,
                                                  SPRITE_UNIT_SIZE])
    centroids.append(centroid)
    centroids_vocab.append(name)
    if not name == "Aggregation":
        candidate = {}
        candidate["name"] = name
        candidate["words"] = words
        aggregation.append(candidate)
    print 'Embedding for', name, 'created'


# CREATE SPRITE TO VISUALISE WORD SENTIMENTS IN TENSORBOARD
def create_sprite(words, vocab):
    # create sprite
    dimension_sprite = int(round(math.sqrt(len(vocab))))
    sprite = Image.new(
        mode='RGBA',
        size=(dimension_sprite * SPRITE_UNIT_SIZE,
              dimension_sprite * SPRITE_UNIT_SIZE),
        color=(255, 255, 255, 0))
    count_x = -1
    count_y = 0
    location_x = 0
    location_y = 0
    for word in vocab:
        if count_x < dimension_sprite - 1:
            count_x = count_x + 1
            location_x = SPRITE_UNIT_SIZE * count_x
        else:
            count_x = 0
            location_x = 0
            count_y = count_y + 1
            location_y = SPRITE_UNIT_SIZE * count_y
        sentiment_circle = int(words[word]["sentiment"] * 127)
        image = Image.new(
            mode='RGBA',
            size=(SPRITE_UNIT_SIZE, SPRITE_UNIT_SIZE),
            color=(255, 255, 255, 0))
        draw = ImageDraw.Draw(image)
        draw.rectangle([(10, 10), (90, 90)],
                       fill=(
                       127 - sentiment_circle,
                       127 + sentiment_circle,
                       127 + sentiment_circle,
                       255))
        draw.rectangle([(20, 20), (80, 80)],
                       fill=(255, 255, 255, 255))
        sprite.paste(image, (location_x, location_y))
    # sprite.show()
    print "Sprite created"
    return sprite


# FIND CENTROID FOR A TENSOR
def create_centroid(W):
    Mean = tf.reduce_mean(W, 0)
    centroid = sess.run(Mean)
    return centroid


# CREATE TENSOR FOR ALL WORDS USED IN ALL JSON FILES
def view_aggregation(pretrained):
    global aggregation
    words = {}
    name = "Aggregation"
    if len(aggregation) > 1:
        for candidate in aggregation:
            for word, item in candidate["words"].items():
                if word not in words:
                    words[word] = {}
                words[word][candidate["name"]] = item["relevance"]
                words[word]["found"] = False
        create_tensor(words, name, pretrained)


# CREATE TENSOR WITH ALL CENTROIDS
def view_centroids():
    global centroids
    name = "Centroids"
    if len(centroids) > 1:
        metadata = open(os.path.join(DIR, DIR_MODEL, name + ".tsv"), 'w')
        for word in centroids_vocab:
            metadata.write(word + "\n")
        metadata.close()
        centroids_size = len(centroids_vocab)
        embedding_dim = len(centroids[0])
        centroids = np.asarray(centroids)
        # create tensor
        W = tf.Variable(tf.constant(0.0,
                                    shape=[centroids_size, embedding_dim]),
                        trainable=False, name=name)
        embedding_placeholder = tf.placeholder(tf.float32,
                                               [centroids_size, embedding_dim])
        embedding_init = W.assign(embedding_placeholder)
        sess.run(embedding_init, feed_dict={embedding_placeholder: centroids})
        print 'Tensor for centroids created'

        # connect with labels
        embedding = config.embeddings.add()
        embedding.tensor_name = W.name
        embedding.metadata_path = os.path.join(DIR, DIR_MODEL, name + ".tsv")
        print 'Embedding for centroids created'


# MAIN SCRIPT
args = get_input(sys.argv[1:])  # read command arguments
DIR = args[0]
pretrained = args[1]
# get all files with json extension, sorted by name
previous_name = '-'
words = {}
for root, dirs, files in os.walk(DIR):
    for name in sorted(files):
        if name.endswith('.json'):
            tensor_name = name.strip(r"-\[01\]\.json")
            previous_tensor_name = previous_name.strip(r"-\[01\]\.json")
            if not previous_name == '-' and not \
               tensor_name == previous_tensor_name:
                print 'Processing', previous_tensor_name, 'data'
                create_tensor(words,
                              previous_tensor_name,
                              pretrained)
                words = {}
            print root, dirs
            print 'Parsing ' + os.path.join(root, name)
            previous_name = name
            file = open(os.path.join(root, name))
            xml = json.load(file)
            words = parse_json(xml, words)
create_tensor(words, previous_tensor_name, pretrained)
view_aggregation(pretrained)
view_centroids()
saver = tf.train.Saver()
saver.save(sess, os.path.join(DIR, DIR_MODEL, "model.ckpt"), 0)
summary_writer = tf.summary.FileWriter(os.path.join(DIR, DIR_MODEL))
projector.visualize_embeddings(summary_writer, config)
