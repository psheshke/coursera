/**
 * @param {Array} collection
 * @params {Function[]} – Функции для запроса
 * @returns {Array}
 */
function query(collection) {
    var newCollection = collection

    console.log('query args: ', arguments.length, newCollection === arguments[0])
    return 8
}

/**
 * @params {String[]}
 */
function select() {
    console.log('select args: ', arguments)
    return ['select', arguments]
}

/**
 * @param {String} property – Свойство для фильтрации
 * @param {Array} values – Массив разрешённых значений
 */
function filterIn(property, values) {
    console.log('filterIn args: ', arguments)
    return arguments
}

module.exports = {
    query: query,
    select: select,
    filterIn: filterIn
};
