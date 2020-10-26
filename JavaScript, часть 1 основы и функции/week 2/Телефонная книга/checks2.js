//После команд "ADD Ivan 555,666; ADD Alex 777; ADD Alex 333; REMOVE_PHONE 555; REMOVE_PHONE 666; ADD Ivan 888; SHOW", ожидается результат: ["Alex: 777, 333","Ivan: 888"]

// Встроенный в Node.JS модуль для проверок
var assert = require('assert');

// Подключаем свою функцию
var phoneBook = require('./index.js');

phoneBook('ADD Ivan 555,666');

phoneBook('ADD Alex 777');

phoneBook('ADD Alex 333');

phoneBook('REMOVE_PHONE 555');

phoneBook('REMOVE_PHONE 666');

phoneBook('ADD Ivan 888');

assert.deepEqual(
    // Получаем содержимое телефонной книги
    phoneBook('SHOW'),
    [
        'Alex: 777, 333',
        'Ivan: 888'
    ],
    'В телефонной книге: "Alex: 777, 333", "Ivan: 888"'
);

console.info('OK!');