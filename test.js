import BrainThreadSafe from './';

const net = new BrainThreadSafe({ hiddenLayers: [4] });

net.train([{input: [0, 0], output: [0]},
  {input: [0, 1], output: [1]},
  {input: [1, 0], output: [1]},
  {input: [1, 1], output: [0]}], {
  log: true,
  logPeriod: 10,
  doneCallback: () => {
    console.log(net.run([1, 0]));
  }
});