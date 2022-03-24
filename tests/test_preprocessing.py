
from src.preprocessing import *

def test_remove_retweets() -> None:

    test = "RT @123jlo_: this is a test"
    res = remove_retweets(test)
    assert res == ' this is a test'

def test_count_retweets() -> None:

    test = "RT @123jlo_: blablabla RT @123jlo_: test"
    res = count_retweets(test)
    assert res == 2

def test_remove_mentions() -> None:

    test = "This is just as test @test123 yes of course @blabla"
    res = remove_mentions(test)
    assert res == 'This is just as test  yes of course '

def test_count_mentions() -> None:

    test = "This is just as test @test123 yes of course @blabla"
    res = count_mentions(test)
    assert res == 2

def test_remove_urls() -> None:

    test = """https://t.co/tRdrrxJskF blablabla
    test https://t.co/tRdrrxJskF yes"""
    res = remove_urls(test)
    assert res == ' blablabla\n    test  yes'

def test_count_urls() -> None:

    test = """https://t.co/tRdrrxJskF blablabla
    test https://t.co/tRdrrxJskF yes"""
    res = count_urls(test)
    assert res == 2

def test_remove_hashtags() -> None:

    test = "this is a test #test #Covid-19"
    res = remove_hashtags(test)
    assert res == 'this is a test test Covid-19'

def test_count_hashtags() -> None:

    test = "this is a test #test #Covid-19 blabla"
    res = count_hashtags(test)
    assert res == 2

def test_remove_additional_space() -> None:

    test = 'blabla  '
    res = remove_additional_space(test)
    assert res == 'blabla '

def test_replace_slash_chars_by_space() -> None:

    test = "blabla\r\n\r\nyes"
    res = replace_slash_chars_by_space(test)
    assert res == 'blabla    yes'

    

