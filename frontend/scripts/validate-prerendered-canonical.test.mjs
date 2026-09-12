import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { isWwwLirieCanonicalHref } from './validate-prerendered-html.mjs';

describe('isWwwLirieCanonicalHref', () => {
  it('accepte le host exact https://www.lirie.ch', () => {
    assert.equal(isWwwLirieCanonicalHref('https://www.lirie.ch/'), true);
    assert.equal(isWwwLirieCanonicalHref('https://www.lirie.ch/contact'), true);
    assert.equal(isWwwLirieCanonicalHref('https://www.lirie.ch:443/'), true);
  });

  it('refuse lookalikes, userinfo, path-as-host et protocoles dangereux', () => {
    assert.equal(isWwwLirieCanonicalHref('https://www.lirie.ch.evil.example/'), false);
    assert.equal(isWwwLirieCanonicalHref('https://www.lirie.ch@evil.example/'), false);
    assert.equal(isWwwLirieCanonicalHref('https://evil.example/www.lirie.ch'), false);
    assert.equal(isWwwLirieCanonicalHref('https://evil-lirie.ch/'), false);
    assert.equal(isWwwLirieCanonicalHref('https://lirie.ch.evil.example/'), false);
    assert.equal(isWwwLirieCanonicalHref('https://evillirie.ch/'), false);
    assert.equal(isWwwLirieCanonicalHref('javascript:alert(1)'), false);
    assert.equal(isWwwLirieCanonicalHref('data:text/html,hi'), false);
    assert.equal(isWwwLirieCanonicalHref('file:///etc/passwd'), false);
    assert.equal(isWwwLirieCanonicalHref('//www.lirie.ch/'), false);
    assert.equal(isWwwLirieCanonicalHref('https://www.lirie.ch:444/'), false);
    assert.equal(isWwwLirieCanonicalHref('http://www.lirie.ch/'), false);
  });
});
