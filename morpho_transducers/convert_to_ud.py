import sys
import re

'''
feature_dict = {"post":"AdpType=Post", "pr":"AdpType=Prep", "f":"Gender=Fem", "m":"Gender=Masc", "nt":"Gender=Neut", "ut":"Gender=Com", "mf":"Gender=Fem,Gender=Masc", "mn":"Gender=Masc,Gender=Neut", "fn":"Gender=Fem,Gender=Neut", "mfn":"Gender=Masc,Gender=Neut,Gender=Fem", "mp":"Gender=Masc|Animacy=Hum", "ma":"Gender=Masc|Animacy=Anim", "mi":"Gender=Masc|Animacy=Inan", "aa":"Animacy=Anim", "an":"Animacy=Anim,Animacy=Inan", "nn":"Animacy=Inan", "pers":"PronType=Prs", "rel":"PronType=Rel", "acr":"Abbr=Yes", "ind":"PronType=Ind", "ref":"Reflex=Yes", "pos":"Poss=Yes", "dem":"PronType=Dem", "def":"PronType=Art|Definite=Def", "itg":"PronType=Int", "ind":"PronType=Art|Definite=Ind", "sg":"Number=Sing", "pl":"Number=Plur", "sp":"Number=Plur,Number=Sing", "du":"Number=Dual", "nom":"Case=Nom", "acc":"Case=Acc", "dat":"Case=Dat", "gen":"Case=Gen", "dg":"Case=Dat,Case=Gen", "voc":"Case=Voc", "abl":"Case=Abl", "ins":"Case=Ins", "loc":"Case=Loc", "tra":"Case=Tra", "ill":"Case=Ill", "ine":"Case=Ine", "ade":"Case=Ade", "all":"Case=All", "abe":"Case=Abe", "ess":"Case=Ess", "par":"Case=Par", "dis":"Case=Dis", "com":"Case=Com", "actv":"Voice=Act", "pass":"Voice=Pass", "pasv":"Voice=Pass", "midv":"Voice=Mid", "caus":"Voice=Cau", "pres":"Tense=Pres", "pret":"Tense=Past", "past":"Tense=Past", "pmp":"Tense=Pqp", "plu":"Tense=Pqp"}
pos_dict = {"n":"NOUN", "np":"PROPN", "vblex":"VERB", "vbser":"COP", "vbhaver":"VERB", "vaux":"AUX", "adj":"ADJ", "post":"ADP", "adv":"ADV", "preadv":"ADV", "postadv":"ADV", "mod":"", "det":"DET", "prn":"PRON", "pr":"ADP", "num":"NUM", "ij":"INTJ", "cncoo":"CCONJ", "cnjsub":"SCONJ", "cnjadv":"ADV,SCONJ", "sent":"PUNCT", "cm":"PUNCT", "lquot":"PUNCT", "rquot":"PUNCT"}
'''

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
	print("foobar")

def apertium_to_conllu(line):
	results = []
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
				print("Tag: '" + t + "' was not recognized.", file=sys.stderr)
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

		print(pos_list)
		print(separated_feature_fields)

		for pos in pos_list:
			for feature_field in separated_feature_fields:
				results.append(analyzed_input + "\t" + lemma + "\t" + pos + "\t" + feature_field)

	for r in results:
		print(r)
	
	print("\n")


input_format = sys.argv[1] # "apertium" or "giella"

pos_dict, feature_dict = load_dictionaries("apertium2ud.tsv")

print("Dictionaries loaded:")
print(pos_dict)
print(feature_dict)

for line in sys.stdin:
	if line and len(line) > 0:
		print("Line: " + line)
		if input_format == "apertium":
			apertium_to_conllu(line)
		elif input_format == "giella":
			giella_to_conllu(line)
		else:
			print("Format missing on unknown. Please use either 'apertium' or 'giella'")
