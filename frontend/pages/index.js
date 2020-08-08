import { useState } from 'react';
import Head from 'next/head';
import axios from 'axios';

import { Textarea, Text, Button, Box } from '@chakra-ui/core';

export default function Home() {
  const [text, setText] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [analyzeResult, setAnalyzeResult] = useState(null);

  const handleTextChange = (e) => {
    setText(e.target.value);
    setAnalyzeResult(null);
  };

  const handleOnClick = () => {
    setIsSubmitting(true);

    // TODO: add the correct endpoint.
    axios
      .post('/predict', {
        text,
      })
      .then(function (response) {
        // assumes that response is a number between 0.0-1.0
        setAnalyzeResult(response.data.result);
      })
      .catch(function (error) {
        setIsSubmitting(false);
        console.log(error);
      });
  };

  return (
    <Box maxW="960px" mx="auto" padding="15px">
      <Head>
        <title>Fake News</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <Text fontSize="5xl" as="h1">
        Welcome to Fake News
      </Text>

      <Text fontSize="xl" as="p">
        Some fancy tagline here
      </Text>

      <Box mt="30px" mb="30px">
        <Textarea
          value={text}
          onChange={handleTextChange}
          placeholder="Type or paste your text here."
          mb="25px"
          height="300px"
        />
        <Button
          variantColor="teal"
          variant="solid"
          isLoading={isSubmitting}
          loadingText="Analyzing..."
          onClick={handleOnClick}
        >
          Analyze text
        </Button>
      </Box>
      {analyzeResult != null && (
        <Box>
          <Text fontSize="lg" as="p">
            The model thinks that the above text is <strong>{analyzeResult * 100}%</strong> fake.
          </Text>
        </Box>
      )}
    </Box>
  );
}
