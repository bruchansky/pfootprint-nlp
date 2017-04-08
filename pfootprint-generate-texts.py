''' SCRIPT TO PARSE HTML FILES AND CREATE TXT FILES
ARGUMENTS:
-d: directory with all your html files.
-t: type, either "debate" or "declaration".
In case of a declaration, each html file generates one text file.
In case of a debate, the script splits html files into
individual contributions (assuming each contribution starts
with the name of the contributor in bold),
and creates one text file per contributor.'''

import sys
import getopt
import codecs
import re
import os.path
import html2text


reload(sys)
sys.setdefaultencoding('utf-8')


def main(argv):
    # GET ARGUMENTS
    directory = ''
    type = 'declaration'
    try:
        opts, args = getopt.getopt(argv, "hd:t:", ["dir=", "type="])
    except getopt.GetoptError:
        print 'pfootprint-generate-texts.py -d <dir> -t <type>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'pfootprint-generate-texts.py -d <dir> -t <type>'
            sys.exit()
        elif opt in ("-d", "--dir"):
            directory = arg
        elif opt in ("-t", "--type"):
            type = arg

    # FOR EACH FILE IN THE DIRECTORY
    for root, dirs, files in os.walk(directory):
        for name in sorted(files):
            with codecs.open(os.path.join(directory, name),
                             encoding='cp1252') as f:
                data = f.read()
                data = html2text.html2text(data)
                data = re.sub(r"\[.*?\]", '', data)
                if type == 'debate':
                    # SPLIT THE FILE IN PARAGRAPHS AND ASSIGN TO CANDIDATES
                    match = re.findall(r'\*\*(.*?):\*\*([^\*]*)',
                                       data, re.DOTALL)
                    for candidate, paragraph in match:
                        try:
                            f = open(os.path.join(directory,
                                     root.split(os.sep)[-2] +
                                     '-' + candidate + '.txt'), "a")
                            f.write(paragraph + "\n")
                            f.close()
                        except Exception:
                            print 'MISSMATCH:', paragraph
                if type == 'declaration':
                    f = open(os.path.join(directory, os.path.splitext(name)[0] + '.txt'), "w")
                    f.write(data)
                    f.close()


if __name__ == "__main__":
    main(sys.argv[1:])
