# super-super-simple-classification


Wanna classify text? 

Seems like you should be able to without learning about some crazy function like  `(m,b)=1N∑i=1n(yi−(mxi+b))2` right! ⬅ btw thats a real formula and is called gradient descent.

Thats what I think too‼️

**super-super-simple-classification** is a project that aims to make this as easy as possible! No math needed, no insights to how the human brain works. Just good ol' repurposing of some of Google's ML research and useful Python libraries. 


## The Model

The powerhouse of this whole thing is what Google calls the "Universal Sentence Encoder Large (3)". This model is a vector encoding model that has been built with Tensorflow and billions of pieces of text that Google has collected. 

Are you a GMail users? Have you noticed the email client attempting to guess the next word(s). Its pretty good right? Thats the same model (a slightly smaller faster one). Assuming you've seen this feature in GMail, you know what this bad boy can do!


## Building your classifications

Now in order to make this useful to you, you'll need to setup some keywords for the classifications your looking for. You'll need to do this in the `config/groups.json` file. 

```json
{
    "pets": [
        "dog", "cat", "pet", "leash", "bone"
    ],
    "colors": [
        "red", "blue", "maroon", "yellow", "green"
    ],
    "electronics": [
        "tv", "car", "toaster", "microwave"
    ]
}
```

That wasn't hard was it! 

Note: You can change this file while the app is running to update your groups on the fly 🛸



### REST endpoints

Just make a simple `POST` resquest with the same format body below and see how similar the semantic meaning of your word is wirh your groups!
```js
fetch("http://localhost:5000/classify", {
  "method": "POST",
  "headers": {
    "content-type": "application/json"
  },
  "body": {
    "a": "paws"
  }
})
.then(response => {
  console.log(response);
})
.catch(err => {
  console.log(err);
});
```


```json
{
  "latency": 260,
  "latency1": 259,
  "summaries": {
    "colors": {
      "max:": 0.4424898624420166,
      "min:": 0.3360968828201294,
      "median:": 0.39735910296440125,
      "n": 5,
      "q3:": 0.42034441232681274,
      "p90:": 0.4336316823959351,
      "range": 0.10639297962188721,
      "data_std": 0.03677679973915049,
      "q1:": 0.3751916289329529
    },
    "electronics": {
      "max:": 0.44020071625709534,
      "min:": 0.3194732666015625,
      "median:": 0.3756273090839386,
      "n": 4,
      "q3:": 0.42150162905454636,
      "p90:": 0.43272108137607573,
      "range": 0.12072744965553284,
      "data_std": 0.05110808122141922,
      "q1:": 0.331857830286026
    },
    "pets": {
      "max:": 0.7221435904502869,
      "min:": 0.5844173431396484,
      "median:": 0.6691290140151978,
      "n": 5,
      "q3:": 0.6965268850326538,
      "p90:": 0.7118969082832336,
      "range": 0.13772624731063843,
      "data_std": 0.051429937071752675,
      "q1:": 0.6122357845306396
    }
  }
```