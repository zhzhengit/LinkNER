import random

def sample_example(dic, shot, label_num):
    data_length = (len(dic))
    example = []
    count = 0
    if shot<=0:
        return []
    else:
        set_example = label_num

    for i in range (shot):
        label = []
        while len(label)<set_example:
            idx = random.randint(0,data_length) - 1
            data = dic[idx]
            if data['answer'] not in label:
                count += 1
                example.append("Example:\n"+data['question']+data['answer']+"\n")
                label.append(data['answer'])
            else:
                pass
    return("".join(example))

def labr(data_name):
    if 'conll03' in data_name or 'wiki' in data_name or 'typos' in data_name or 'oov' in data_name:
        label_describe =  "Here is the entity class information: \
        Person: This category includes names of persons, such as individual people or groups of people with personal names. \
        Organization: The organization category consists of names of companies, institutions, or any other group or entity formed for a specific purpose. \
        Location: The location category represents names of geographical places or landmarks, such as cities, countries, rivers, or mountains. \
        Miscellaneous: The miscellaneous category encompasses entities that do not fall into the above three categories. This includes adjectives, like Italian, and events, like 1000 Lakes Rally, making it a very diverse category. Examples of entity classification are given below:"
        label = "\n Location\n Person\n Organization\n Miscellaneous\n Non-entity"
        label_set = {"Location":'LOC',"Person":'PER',"Organization":'ORG',"Miscellaneous":'MISC',"Non-entity":'Non'}
    elif 'wnut17' in data_name:
    # wnut17 labels
        label_describe = "A dataset focus on unusual, previous-unseen entities in example data, and is collected from social media.\
        In the test example,  we included Twitter as a source, but additional comments were mined from Reddit, YouTube and StackExchange. Here is all entity class information:\
        Person: Names of people (e.g. Virginia Wade). Don’t mark people that don’t have their own name. Include punctuation in the middle of names. Fictional people can be included, as long as they’re referred to by name (e.g. Harry Potter).\
        Location: Names that are locations (e.g. France). Don’t mark locations that don’t have their own name. Include punctuation in the middle of names. Fictional locations can be included, as long as they’re referred to by name (e.g. Hogwarts).\
        Corporation: Names of corporations (e.g. Google). Don’t mark locations that don’t have their own name. Include punctuation in the middle of names.\
        Product: Name of products (e.g. iPhone). Don’t mark products that don’t have their own name. Include punctuation in the middle of names. Fictional products can be included, as long as they’re referred to by name (e.g. Everlasting Gobstopper). It’s got to be something you can touch, and it’s got to be the official name.\
        Creative-work: Names of creative works (e.g. Bohemian Rhapsody). Include punctuation in the middle of names. The work should be created by a human, and referred to by its specific name.\
        Group: Names of groups (e.g. Nirvana, San Diego Padres). Don’t mark groups that don’t have a specific, unique name, or companies (which should be marked corporation)."
        label = "\n Location\n Group\n Corporation\n Person\n Creative-work\n Product\n Non-entity"
        label_set = {
            "Location":'location', 
            "Group":'group',
            "Corporation":'corporation',
            "Person":'person',
            "Creative-work":'creative-work',
            "Product":'product',
            "Non-entity":'Non'}
   
    elif 'Ontonote' in data_name:
    # Ontonote 5 labels
        label_describe = "Here is the entity class information: \
        PERSON: People, including fictional\
        ORGANIZATION: Companies, agencies, institutions, etc.\
        GPE: Countries, cities, states \
        DATE: Absolute or relative dates or periods\
        NORP: Nationalities or religious or political groups \
        CARDINAL: Numerals that do not fall under another type\
        TIME: Times smaller than a day \
        LOC: Non-GPE locations, mountain ranges, bodies of water \
        FACILITY:Buildings, airports, highways, bridges, etc. \
        PRODUCT: Vehicles, weapons, foods, etc. (Not services) \
        WORK_OF_ART: Titles of books, songs, etc. \
        MONEY: Monetary values, including unit \
        ORDINAL: “first”, “second”, etc. \
        QUANTITY: Measurements, as of weight or distance, etc. \
        EVENT: Named hurricanes, battles, wars, sports events, etc. \
        PERCENT: Percentage (including “%”), etc. \
        LAW: Named documents made into laws, etc. \
        LANGUAGE: Any named language, etc. "
        label = "\n Person\n Organization\n GPE\n Date\n NORP\n Cardinal\n Time\n Location\n Facility\n Product\n WORK_OF_ART\n Money\n Ordinal\n Quantity\n Event\n Percent\n Law\n Language\n Non-entity"
        label_set = {
            "Person": 'PERSON',
            "Organization": 'ORG',
            "GPE": 'GPE',
            "Date": 'DATE',
            "NORP": 'NORP',
            "Cardinal": 'CARDINAL',
            "Time": 'TIME',
            "Location": 'LOC',
            "Facility": 'FAC',
            "Product": 'PRODUCT',
            "WORK_OF_ART": 'WORK_OF_ART',
            "Money": 'MONEY',
            "Ordinal": 'ORDINAL',
            "Quantity": 'QUANTITY',
            "Event": 'EVENT',
            "Percent": 'PERCENT',
            "Law": 'LAW',
            "Language": 'LANGUAGE',
            "Non-entity": 'NON'
        }    
    elif 'Twitter' in data_name:
    # Twitter labels
        label_describe =  "Here is the entity class information: For polysemous entities, our guidelines instructed annotators to assign the entity class that corresponds to the correct entity class in the given context. For example, in “We’re driving to Manchester”, Manchester is a Location, but in “Manchester are in the final tonight”, it is a sports club – an Organization.\
        Special attention is given to username mentions. Where other corpora have blocked these out or classified them universally as Person, our approach is to treat these as named entities of any potential class. For example, the account belonging to the \"Manchester United football club\" would be labeled as an Organization. Other: The Other category encompasses entities that do not fall into the above three categories. This includes adjectives, like Italian, and events, like 1000 Lakes Rally, making it a very diverse category. Examples of entity classification are given below:"
        label = "\n Location\n Person\n Organization\n Other\n Non-entity"
        label_set = {"Person":'PER', "Location":'LOC', "Other":'OTHER', "Organization":'ORG', "Non-entity":'Non'}
    else:
        raise Exception("Invalid dataname! Please check")
    
    label_num = len(label_set)-1
    
    return label_describe, label, label_set, label_num
