module.exports = {

    events: new Array(),

    /**
     * @param {String} event
     * @param {Object} subscriber
     * @param {Function} handler
     */
    on: function (event, subscriber, handler) {
        this.events.push({event: event, subscriber: subscriber, handler: handler});
        return this;
    },

    /**
     * @param {String} event
     * @param {Object} subscriber
     */
    off: function (event, subscriber) {
        for (let i = this.events.length - 1; i >= 0; i--) {
            let curEvent = this.events[i];
            if (curEvent.event == event && curEvent.subscriber == subscriber) {
                this.events.splice(i, 1);
            }
        }
        return this;
    },

    /**
     * @param {String} event
     */
    emit: function (event) {
        for (let i in this.events) {
            if (this.events[i].event == event) {
                this.events[i].handler.call(this.events[i].subscriber);
            }
        }
        return this;
    }
};