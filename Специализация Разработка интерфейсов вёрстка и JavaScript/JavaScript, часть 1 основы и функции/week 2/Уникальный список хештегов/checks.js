// Встроенный в Node.JS модуль для проверок
var assert = require('assert');

// Подключаем свою функцию
var normalizeHashTags = require('./index.js');

assert.deepEqual(
    normalizeHashTags(['web', 'coursera', 'JavaScript', 'Coursera', 'script', 'programming']),
    'web, coursera, javascript, script, programming',
    'Список "web, coursera, JavaScript, Coursera, script, programming"' +
    ' содержит хэштеги "web, coursera, javascript, script, programming"'
);

assert.deepEqual(
    normalizeHashTags(['JavaScript', 'web', 'WEB']),
    'javascript, web',
    'Список "JavaScript, web, WEB"' +
    ' содержит хэштеги "javascript, web"'
);

assert.deepEqual(
    normalizeHashTags([]),
    '',
    'Список ""' +
    ' содержит хэштеги ""'
);

console.info('OK!');
