import brain from 'brain.js';
import Thaw from 'thaw.js';

export default class BrainThreadSafe extends brain.NeuralNetwork {
  /**
   *
   * @param data
   * @param _options
   * @returns {{error: number, iterations: number}}
   */
  train(data, _options = {}) {
    const options = Object.assign({}, this.constructor.trainDefaults, _options);
    data = this.formatData(data);
    let iterations = options.iterations;
    let errorThresh = options.errorThresh;
    let log = options.log === true ? console.log : options.log;
    let logPeriod = options.logPeriod;
    let learningRate = _options.learningRate || this.learningRate || options.learningRate;
    let callback = options.callback;
    let callbackPeriod = options.callbackPeriod;
    let doneCallback = options.doneCallback;
    if (!options.reinforce) {
      let sizes = [];
      let inputSize = data[0].input.length;
      let outputSize = data[0].output.length;
      let hiddenSizes = this.hiddenSizes;
      if (!hiddenSizes) {
        sizes.push(Math.max(3, Math.floor(inputSize / 2)));
      } else {
        hiddenSizes.forEach(size => {
          sizes.push(size);
        });
      }

      sizes.unshift(inputSize);
      sizes.push(outputSize);

      this.initialize(sizes);
    }

    let error = 1;
    let i = 0;
    
    const items = new Array(iterations);
    const thaw = new Thaw(items, {
      delay: true,
      each: () => {
        i++;
        let sum = 0;
        for (let j = 0; j < data.length; j++) {
          let err = this.trainPattern(data[j].input, data[j].output, learningRate);
          sum += err;
        }
        error = sum / data.length;
  
        if (log && (i % logPeriod === 0)) {
          log('iterations:', i, 'training error:', error);
        }
        if (callback && (i % callbackPeriod === 0)) {
          callback({ error: error, iterations: i });
        }
        if (error < errorThresh) {
          thaw.stop();
        }
      },
      done: () => {
        if (doneCallback) doneCallback({
          error: error,
          iterations: i
        });
      }
    });
    thaw.tick();
  }
}