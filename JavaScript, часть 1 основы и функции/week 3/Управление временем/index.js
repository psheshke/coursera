/**
 * @param {String} date
 * @returns {Object}
 */

var validUnits = ['years', 'months', 'days', 'hours', 'minutes']

module.exports = function (date) {
    return {
        time: new Date(date),
        get value() {
            var year = this.time.getFullYear();
            var month = this.time.getMonth();
            var day = this.time.getDate();
            var hours = this.time.getHours();
            var minutes = this.time.getMinutes();
            return year + '-' + ('0' + (month + 1)).slice(-2) + '-' + ('0' + day).slice(-2) + ' ' + ('0' + hours).slice(-2) + ':' + ('0' + minutes).slice(-2)
        },
        add: function(num, unit) {
            if (num < 0) {
                throw new TypeError('Передано отрицательное значение');
            }
            if (validUnits.includes(unit) === false) {
                throw new TypeError('Передан неверный параметр');
            }
            if (unit === 'years') {
                this.time.setFullYear(this.time.getFullYear() + num)
            }
            if (unit === 'months') {
                this.time.setMonth(this.time.getMonth() + num)
            }
            if (unit === 'days') {
                this.time.setDate(this.time.getDate() + num)
            }
            if (unit === 'hours') {
                this.time.setHours(this.time.getHours() + num)
            }
            if (unit === 'minutes') {
                this.time.setMinutes(this.time.getMinutes() + num)
            }
            return this;
        },
        subtract: function(num, unit) {
            if (num < 0) {
                throw new TypeError('Передано отрицательное значение');
            }
            if (validUnits.includes(unit) === false) {
                throw new TypeError('Передан неверный параметр');
            }
            if (unit === 'years') {
                this.time.setFullYear(this.time.getFullYear() - num)
            }
            if (unit === 'months') {
                this.time.setMonth(this.time.getMonth() - num)
            }
            if (unit === 'days') {
                this.time.setDate(this.time.getDate() - num)
            }
            if (unit === 'hours') {
                this.time.setHours(this.time.getHours() - num)
            }
            if (unit === 'minutes') {
                this.time.setMinutes(this.time.getMinutes() - num)
            }
            return this;
        }
    };
};
