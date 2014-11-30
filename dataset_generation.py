#!/usr/bin/env python3

import itertools

person = [
    "Obama",
    "Barack Obama",
    "Barack Hussein Obama",
    "Lennon",
    "John Lennon",
    "Saint-Exupéry",
    "Antoine de Saint-Exupéry",
    "Antoine Marie Jean-Baptiste Roger de Saint-Exupéry"
]

country = [
    "United States",
    "United States of America",
    "USA",
    "US",
    "U.S.A.",
    "U.S.",
    "United Kingdom",
    "UK",
    "U.K.",
    "France",
    "North Korea",
    "Saudi Arabia",
    "New Zealand"
]

city = [
    "Paris",
    "New York",
    "San Francisco",
    "Los Angeles",
    "Ho Chi Minh City",
    "New Delhi"
]

location = country+city+[
    "South America",
    "Middle East",
    "Africa",
    "Pacific Ocean",
    "Mediterranean Sea",
    "America",
    "Europe",
    "Asia",
    "Oceania"
]

book = [
    "A Tale of Two Cities",
    "The Lord of the Rings",
    "Le Petit Prince",
    "Harry Potter and the Philosopher's Stone",
    "And Then There Were None",
    "Dream of the Red Chamber",
    "The Hobbit",
    "She: A History of Adventure"
]

film = [
    "Avatar",
    "Titanic",
    "The Avengers",
    "Harry Potter and the Deathly Hallows",
    "Frozen",
    "Iron Man 3",
    "Transformers: Dark of the Moon",
    "The Lord of the Rings: The Return of the King"
]

single = [
    "White Christmas",
    "I Will Always Love You",
    "We Are the World",
    "Da Da Da",
    "Hey Jude",
    "Bohemian Rhapsody"
]

art = book+film+single

def print_data(subject,predicate):
    triple = "\n{0} | {1} | _".format(subject," ".join(predicate))
    for perm in itertools.permutations(predicate):
        for i in range(0,len(perm)+1):
            sentence = perm[0:i]+(subject,)+perm[i:len(perm)]
            print(" ".join(sentence+(triple,)) + "\n")
            print((" ".join(sentence+(triple,)) + "\n").lower())

def print_person():
    for p in person:
        for ev in {"death","birth"}:
            for obj in {"place","date"}:
                print_data(p,[obj,ev])

def print_country():
    for c in country:
        print_data(c,["president"])
        print_data(c,["prime", "minister"])

def print_city():
    for c in city:
        print_data(c,["mayor"])

def print_location():
    for l in location:
        print_data(l,["population"])

def print_film():
    for f in film:
        print_data(f,["cast","member"])
        print_data(f,["director"])

def print_book():
    for b in book:
        print_data(b,["original","language"])
        print_data(b,["author"])

def print_single():
    for s in single:
        print_data(s,["record","label"])

def print_art():
    for a in art:
        print_data(a,["official","website"])
        print_data(a,["date","publication"])

def print_all():
    print_person()
    print_country()
    print_city()
    print_location()
    print_film()
    print_book()
    print_single()
    print_art()

print_all()
