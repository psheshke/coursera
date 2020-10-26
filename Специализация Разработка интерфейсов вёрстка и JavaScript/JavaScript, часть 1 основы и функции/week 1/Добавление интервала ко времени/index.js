/**
 * @param {Number} hours
 * @param {Number} minutes
 * @param {Number} interval
 * @returns {String}
 */
module.exports = function (hours, minutes, interval) {

    hours = (hours + ((minutes + interval) - (minutes + interval) % 60) / 60) % 24

    minutes = (minutes + interval) % 60

    return ('0' + hours).slice(-2) + ':' + ('0' + minutes).slice(-2)
};
