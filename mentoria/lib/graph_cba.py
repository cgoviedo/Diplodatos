from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import seaborn

def plot_word_frequency(axes , x , y , df , title , limit_to_plot ):

    seaborn.barplot(ax = axes , x=x, y=y, data=df.sort_values(y ,ascending=False)[0:limit_to_plot])
    axes.set_xticklabels(axes.get_xticklabels(),rotation=90)
    axes.set_xlabel('Palabras')
    axes.set_ylabel('Ocurrencias')
    axes.set_title(title)



def plot_word_cloud (axes, df, limit_words, title , columns):

    tuples = [tuple(x) for x in df[columns].values]
    wordcloud = WordCloud(background_color="white" , max_words=limit_words).generate_from_frequencies(dict(tuples))

    axes.imshow(wordcloud, interpolation='bilinear')
    axes.axis('off')
    axes.set_title(title)
