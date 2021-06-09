---
permalink: /
title: "About me"
excerpt: "About me"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

I'm a Ph.D. student of the EDIC department at [EPFL](https://www.epfl.ch/en/), and research assistant at [IDIAP research institute](https://www.idiap.ch/en). I'm a member of the [Natural Language Understanding](https://www.idiap.ch/en/scientific-research/natural-language-understanding) group under the supervision of [Dr. James Henderson](https://www.idiap.ch/~jhenderson/). I received my bachelor's degree in electrical engineering from [Sharif University of Technology](http://www.sharif.ir/home) (also minor in computer science).    



I'm currently working on applying deep learning models on natural languages, specifically representation learning and parsing, graph encoding, and structure prediction. We propose a general structured prediction model called [Graph-to-Graph Transformers]({{ site.url }}/publications) to encode and predict arbitrary graphs. Our architecture can be applied to many NLP tasks. Also, I am the member of [Intrepid Project](https://www.intrepid-project.com/), funded by the Swiss National Science Foundation, which aims to develop a general understanding of how policy announcements by state agencies are interpreted by journalists in ways that send signals, indicate intent, and otherwise provoke economic and political reactions.

<!-- I am keen to initiate any kind of academic collaborations, so if you have similar research interests, please feel free to drop me a message! -->

# News

{% include base_path %}
{% capture written_year %}'None'{% endcapture %}
{% for post in site.posts  limit:3  %}
  {% capture year %}{{ post.date | date: '%Y' }}{% endcapture %}
  {% include archive-single.html %}
{% endfor %}

### [See more...]({{ site.url }}/updates)
