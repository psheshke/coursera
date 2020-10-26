/**
 * @param {String[]} hashtags
 * @returns {String}
 */
module.exports = function (hashtags) {

    var result = [];

    for (var i = 0; i < hashtags.length; i++) {
        var tag = hashtags[i];

        var F = true;

        for (var j = 0; j < result.length; j++) {
            if (tag.toLowerCase() == result[j].toLowerCase()) {
                F = false;
            };
        };

        if (F) {
            result.push(tag.toLowerCase());
        };
    };

    return result.join(', ');

};

