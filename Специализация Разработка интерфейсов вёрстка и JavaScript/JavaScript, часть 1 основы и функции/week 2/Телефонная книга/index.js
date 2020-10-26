// Телефонная книга
var phoneBook = {};

/**
 * @param {String} command
 * @returns {*} - результат зависит от команды
 */
module.exports = function (command) {

    var commandName = command.split(' ')[0];

    if (commandName === 'ADD') {

        var contactName = command.split(' ')[1];

        var phoneNumbers = command.split(' ')[2].split(',');

        if (phoneBook.hasOwnProperty(contactName) === true) {
            phoneBook[contactName] = phoneBook[contactName].concat(phoneNumbers);
        } else {
            phoneBook[contactName] = phoneNumbers;
        }
        return undefined;
    } else if (commandName === 'SHOW') {
        var showResult = [];

        var keys = Object.keys(phoneBook).sort();

        for (var i = 0; i < keys.length; i++) {
            var key = keys[i];

            showResult.push(key + ': ' + phoneBook[key].join(', '));
        }
        return showResult;

    } else if (commandName === 'REMOVE_PHONE') {
        var phoneNumberForRemove = command.split(' ')[1];
        var keys = Object.keys(phoneBook);

        for (var i = 0; i < keys.length; i++) {
            var removeResult = false;
            var elementForRemove = 0;
            var key = keys[i];
            for (var j = 0; j < phoneBook[key].length; j++) {
                if (phoneNumberForRemove === phoneBook[key][j]) {
                    removeResult = true;
                    elementForRemove = j;
                }
            }
            if (removeResult !== false) {
                phoneBook[key].splice(elementForRemove, 1)
            }

            if (phoneBook[key].length === 0) {
                delete phoneBook[key];
            }
        }
        return removeResult;
    }

};

function addPhone() {

};