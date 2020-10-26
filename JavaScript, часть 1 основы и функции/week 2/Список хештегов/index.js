/**
 * @param {String} tweet
 * @returns {String[]}
 */
module.exports = function (tweet) {

    var result = [];

    var tweets = tweet.split(' ');

    result = tweets.filter(filterWithWstdaysHashtag).map(dropHashtag);

    return result;

};

function filterWithWstdaysHashtag(tweet, index) {
    return tweet.indexOf('#') !== -1;
};

function dropHashtag(tweet) {
    return tweet.slice(1, tweet.length)
};