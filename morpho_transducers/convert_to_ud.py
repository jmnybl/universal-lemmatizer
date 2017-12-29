import sys
import re
import argparse

parser = argparse.ArgumentParser("This script converts POS- and feature-tags from other formats to CoNLL-U")
parser.add_argument("-v", "--verbose", help="prints debug outputs", action="store_true")
parser.add_argument("-f", "--format", help="define input format", type=str, choices=["apertium", "giella"], required=True)
args = parser.parse_args()



def load_dictionaries(filepath):
	with open(filepath, "rt") as f:
		pos_dict = {}
		feature_dict = {}
		for line in f:
			if line and not line.startswith("#"):
				split = line.split("\t")
				origin = ""
				u_pos = split[1]
				morpho = split[2]
				new_pos = split[3]
				new_morpho = split[4]
				if u_pos != "_":
					origin = u_pos
				elif morpho != "_":
					origin = morpho
				if origin and new_pos != "_":
					pos_dict[origin] = new_pos
				if origin and new_morpho != "_":
					feature_dict[origin] = new_morpho
		return pos_dict, feature_dict

# generate separate analysis for each translation induced separate features (denoted by comma-separated result) 
def separate_feature_analysis(feature_list):
	return_list = []
	for feature in feature_list:
		split_list = feature.split(",")
		overwriting_list = []
		for split in split_list:
			if return_list:
				for return_value in return_list:
					overwriting_list.append(return_value + "|" + split)
			else:
				overwriting_list.append(split)
		return_list = overwriting_list
	return return_list


def giella_to_conllu(line):
	results = []
	tab_split = line.split("\t")
	analyzed_input = tab_split[0]
	compound_split = tab_split[1].split("#")
	lemma = ""
	for part in compound_split:
		lemma = lemma + "#" + part.split("+")[0]
	lemma = lemma.strip("#")

	pos_tags = []
	feature_tags = []
	last_tags = compound_split[-1].split("+")
	last_tags.pop(0)
	for tag in last_tags:
		pos_tag = pos_dict.get(tag, "")
		if pos_tag:
			pos_tags.append(pos_tag)
		feature_tag = feature_dict.get(tag, "")
		if feature_tag:
			feature_tags.append(feature_tag)

	pos_field = "_"
	if len(pos_tags) > 0:
		pos_field = pos_tags[0]

	separated_feature_fields = separate_feature_analysis(feature_tags)
	if len(separated_feature_fields) == 0:
		separated_feature_fields.append("_")

	pos_list = pos_field.split(",")
	print("POS-tags: ", end="\t", file=sys.stderr)
	print(pos_list,file=sys.stderr)
	print("Feature-tags: ", end="\t", file=sys.stderr)
	print(separated_feature_fields,file=sys.stderr)

	for pos in pos_list:
		for feature_field in separated_feature_fields:
			features = feature_field.split("|")
			features.sort()
			feature_field = "|".join(features)
			results.append(analyzed_input + "\t" + lemma + "\t" + pos + "\t" + feature_field)

	for r in results:
		print(r)

def apertium_to_conllu(line):
	erroneus_input = re.sub("\^.+?\$", "", line) # cases where apertium transducer has preserved unanalyzed characters between compound words from the raw data
	if erroneus_input:
		print("Erroneus input, discarding.", file=sys.stderr)
		print("\n", file=sys.stderr)
		return ""
	if "*" in line: #apertium notation for unrecognized words
		print("'*' input unrecognized, discarding.", file=sys.stderr)
		print("\n", file=sys.stderr)
		return ""
	results = []

	separated_compounds = re.findall("\^(.*?)\$", line)

	if len(separated_compounds) > 1:
		lemma = ""
		analyzed_input = ""
		for part in separated_compounds:
			split = part.split("/")
			analyzed_input = analyzed_input + split[0].strip("^")
			lemma = lemma + "<>+" + split[1][:split[1].index("<")]
		last_part = separated_compounds.pop()
		last_tags = last_part[last_part.index("<"):]
		print("Found '$^'-separated compound word, line: " + line, file=sys.stderr)
		line = analyzed_input + "/" + lemma.strip("<>+") + last_tags
		print("Generated artificial input: " + line, file=sys.stderr)
		

	split = line.split("/")
	analyzed_input = split.pop(0).strip("^")
	for s in split:
		compound_split = s.split("+")
		tag_start_point = compound_split[-1].index("<") if "<" in compound_split[-1] else len(compound_split[-1]) - 1
		last_features = compound_split[-1][tag_start_point:]

		lemma = compound_split.pop(0)
		tag_start_point = lemma.index("<") if "<" in lemma else len(lemma) - 1
		lemma = lemma[0:tag_start_point]
		
		for part in compound_split:
			tag_start_point = part.index("<") if "<" in part else len(part) - 1
			lemma = lemma + "#" + part[0:tag_start_point]

		old_tags = re.findall("<(.*?)>", last_features)
		pos_tags = []
		feature_tags = []
		for t in old_tags:
			tag_pos = pos_dict.get(t, "")
			tag_feature = feature_dict.get(t, "")
			if not (tag_pos or tag_feature):
				print("Tag: '" + t + "' was not recognized in ", s, file=sys.stderr)
			else:
				if tag_pos:
					pos_tags.append(tag_pos)
				if tag_feature:
					feature_tags.append(tag_feature)
		pos_field = "_"
		if len(pos_tags) > 0:
			pos_field = pos_tags[0]

		separated_feature_fields = separate_feature_analysis(feature_tags)

		if len(separated_feature_fields) == 0:
			separated_feature_fields.append("_")

		pos_list = pos_field.split(",")
		print("POS-tags: ", end="\t", file=sys.stderr)
		print(pos_list,file=sys.stderr)
		print("Feature-tags: ", end="\t", file=sys.stderr)
		print(separated_feature_fields,file=sys.stderr)

		for pos in pos_list:
			for feature_field in separated_feature_fields:
				features = feature_field.split("|")
				features.sort()
				feature_field = "|".join(features)
				results.append(analyzed_input + "\t" + lemma + "\t" + pos + "\t" + feature_field)

	for r in results:
		print(r)
	print("\n")

input_format = args.format # "apertium" or "giella"

if not args.verbose:
	sys.stderr = open("conversion_errors.txt", "w")

pos_dict, feature_dict = load_dictionaries("apertium2ud.tsv")

#print("Dictionaries loaded:",file=sys.stderr)
#print(pos_dict,file=sys.stderr)
#print(feature_dict,file=sys.stderr)

for line in sys.stdin:
	if line == "\n":
		print(line)
	line = line.strip()
	if line and len(line) > 0:
		print("Line: " + line,file=sys.stderr)
		if input_format == "apertium":
			apertium_to_conllu(line)
		elif input_format == "giella":
			giella_to_conllu(line)
